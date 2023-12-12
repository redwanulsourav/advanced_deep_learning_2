import torch
import argparse
from torch.nn.functional import cross_entropy, softmax
from dataset import TextDataset2
from dataset import TextDataset2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import cross_entropy, softmax
import torch
import json
from train import *
feature_size = 384
context_size = 64
head_count = 6
block_count = 6

def decode(encoded_str):
    with open('decode_dict.json','r') as f:
        data = f.read()
    decode_dict = json.loads(data)
    decoded_chars = []
    for x in encoded_str:
        decoded_chars.append(decode_dict[str(x)])
    return ''.join(decoded_chars)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-w','--weights', help='Path to weight file', required=True)
    ap.add_argument('-c','--count', help='Number of characeters to generate', required=True)

    ap = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(ap.weights, map_location=device)
    model.eval()
    model.to(device)
    
    if int(ap.count) != -1:
        x = torch.zeros((1,1), dtype=torch.long).to(device)
        print(decode(model.generate(x, int(ap.count))[0].tolist()))


    

    