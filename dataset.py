import os
from torch.utils.data import Dataset
import pandas as pd
import torch

class TextDataset2(Dataset):
    def __init__(self, context_size=8):
        self.chrset = set()
        self.index = [] # A list of dictionaries, containing the row and offset of the text.
        # self.df = pd.read_csv(f'bitcoin_articles.csv')
        self.context_size = context_size
        f = open(f'internet_archive_scifi_v3.txt')
        self.contents = f.read()
        f.close()
        self.textlen = len(self.contents)
        
        for ch in self.contents:
            self.chrset.add(ch)

        self.encode_dict = {}
        self.decode_dict = {}
        
        for i, x in enumerate(self.chrset):
            self.encode_dict[x] = i
            self.decode_dict[i] = x

    def __len__(self):
        return self.textlen - (self.context_size + 1)
    
    def encode(self, x):
        encoded = [self.encode_dict[ch] for ch in x]
        return encoded
    
    def decode(self, x):
        decode = [self.decode_dict[i] for i in x]
        return ''.join(decode)
    

    def __getitem__(self, idx):
        x = self.contents[idx:idx+self.context_size]
        y = self.contents[idx+1: idx + self.context_size+1]
        assert(len(x) == len(y)), f'Len of x: {len(x)} != Len of y: {len(y)}'

        return (torch.Tensor(self.encode(x)).long(), torch.Tensor(self.encode(y)).long())
    
    def get_chr_set(self):
        return self.chrset