import torch
from train import MyModel
import argparse
from torch.nn.functional import cross_entropy, softmax
from dataset2 import TextDataset2

context_size = 64
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-w','--weights', help='Path to weight file', required=True)
    ap.add_argument('-c','--count', help='Number of characeters to generate', required=True)

    ap = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextDataset2(context_size=64)
    model = MyModel(len(dataset.encode_dict), 64)
    model.load_state_dict(torch.load(ap.weights))
    model.to(device)
    if int(ap.count) != -1:
        x = torch.zeros((1,1), dtype=torch.long).to(device)
        print(dataset.decode(model.generate(x)[0].tolist()))


    

    