from dataset2 import TextDataset2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import cross_entropy, softmax
import torch

class SingleHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_key_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear_query_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear_value_mapping = nn.Linear(in_features=in_dim, out_features=out_dim)
        
    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        key = self.linear_key_mapping(x)    # (B, 8, 32)
        # print(key.shape)
        query = self.linear_query_mapping(x)    # (B, 8, 32)
        value = self.linear_value_mapping(x)
        # print(f'key shape: {key.shape} query shape: {query.shape} value shape: {value.shape}')
        transformation = query @ key.permute(0,2,1) * key.shape[2] ** -0.5  # (B, 8, 32) @ (B, 32, 8) -> (B, 8, 8)
        # print(f'transformation.shape: {transformation.shape}')
        lower_triangular = torch.tril(torch.ones(x.shape[1], x.shape[1])).to(device)
        transformation = transformation.masked_fill(lower_triangular == 0, float('-inf'))
        transformation = softmax(transformation, dim=-1)
        # print(transformation.shape)
        # print(value.shape)
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
        # print(single_head_size)
        for i in range(head_count):
            self.heads.append(SingleHead(feature_size, single_head_size))
        self.heads = nn.Sequential(*self.heads)
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
        # self.relu3 = nn.ReLU()
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
        # x = self.block2(x)
        # x = self.block3(x)
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
            # print(x.shape)
        return total
