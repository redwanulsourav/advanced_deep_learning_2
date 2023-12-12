from dataset import TextDataset2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import cross_entropy, softmax
import torch
import json

feature_size = 384
context_size = 64
head_count = 6
block_count = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_key_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear_query_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear_value_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.register_buffer("lower_triangular", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        key = self.linear_key_mapping(x)
        query = self.linear_query_mapping(x)
        value = self.linear_value_mapping(x)
        transformation = query @ key.permute(0,2,1) * key.shape[2] ** -0.5  # (B, 8, 32) @ (B, 32, 8) -> (B, 8, 8)
        transformation = transformation.masked_fill(self.lower_triangular[:x.shape[1], :x.shape[1]] == 0, float('-inf'))
        transformation = softmax(transformation, dim=-1)
        x = transformation @ value
        return x
class MultiHead(nn.Module):
    def __init__(self, head_count, single_head_size):
        super().__init__()
        self.head_count = head_count
        self.single_head_size = single_head_size
        self.linear4 = nn.Linear(feature_size, feature_size)
        self.dropout = nn.Dropout(0.2)
        self.heads = []
        for i in range(head_count):
            self.heads.append(SingleHead(feature_size, single_head_size))
        self.heads = nn.ModuleList(self.heads)
    def forward(self, x):
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        
        result = torch.cat(outputs, dim=-1)
        result = self.linear4(result)
        result = self.dropout(result)
        return result

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(feature_size, 4*feature_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(4*feature_size, feature_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(feature_size, feature_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.dropout(x)
        
        return x

class SingleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHead(head_count, feature_size//head_count)
        self.feed_forward = MultiLayerPerceptron()
        self.layer_norm1 = nn.LayerNorm(feature_size)
        self.layer_norm2 = nn.LayerNorm(feature_size)
    def forward(self, x):
        x = self.layer_norm1(x + self.multi_head_attention(x))
        x = self.layer_norm2(x + self.feed_forward(x))
        return x

class MyModel(nn.Module):
    def __init__(self, chr_count, context_size):
        super().__init__()
        self.character_embeddings = nn.Embedding(chr_count, feature_size)
        self.position_embeddings = nn.Embedding(context_size, feature_size)
        self.blocks = []
        for i in range(block_count):
            self.blocks.append(SingleBlock())
        self.blocks = nn.Sequential(*self.blocks)
        self.indices_layer = nn.Linear(feature_size, chr_count)
        self.context_size = context_size
    
    def forward(self, x):
        x_1 = self.character_embeddings(x)
        x_2 = self.position_embeddings(torch.arange(0,x.shape[1],1).to(device))

        x = x_1 + x_2
        x = self.blocks(x)
        y_ = self.indices_layer(x)
        return y_

    def generate(self, x, token_count):
        total = x
        for _ in range(token_count):
            x = x.long()
            y_ = self.forward(x)
            y_ = y_[:,-1,:]
            probs = softmax(y_, dim=-1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            x = torch.cat((x, idx_next), dim=1)
            total = torch.cat((total, idx_next), dim=1)
            x = x[0, x.shape[1]-context_size:].unsqueeze(0)
        return total

if __name__ == '__main__':
    dataset = TextDataset2(context_size=context_size)
    
    generator1 = torch.Generator().manual_seed(42)
    train_set, validation_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], generator=generator1)
    training_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False)
    model = torch.load(f'best3.pt') 
    model.to(device)   
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    count = 0
    training_loss_sum = 0
    val_iter = iter(validation_loader)
    for x, y in training_loader:
        count += 1
        if count == 1000:
            break
        model.train()
        x = x.to(device)
        y = y.to(device)
        y_ = model.forward(x)
        y = y.view(-1)
        y_ = y_.view(y_.shape[0]*y_.shape[1], y_.shape[2])
        loss = cross_entropy(y_,y)
        training_loss_sum += loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        optimizer.step()


        if count % 100 == 0:
            model.eval()
            count2 = 0
            loss_sum = 0
            for x, y in validation_loader:
                count2 += 1
                if count2 == 50:
                    break
                x = x.to(device)
                y = y.to(device)
                y_ = model.forward(x)
                y = y.view(-1)
                y_ = y_.view(y_.shape[0]*y_.shape[1], y_.shape[2])
                loss = cross_entropy(y_,y)
                loss_sum += loss
            loss_sum /= 100
            training_loss_sum /= 100
            print(f'training loss: {training_loss_sum} validation loss: {loss_sum}')
            training_loss_sum = 0
            model.train()
    model.eval()
    model.to(torch.device('cpu'))
    torch.save(model, f'best3.pt')
    print(dataset.decode(model.generate(torch.zeros((1,1), dtype=torch.long).to(device), token_count=500)[0].tolist()))
    f = open('encode_dict.json','w')
    f.write(json.dumps(dataset.encode_dict))
    f.close()

    f = open('decode_dict.json','w')
    f.write(json.dumps(dataset.decode_dict))
    f.close()


        