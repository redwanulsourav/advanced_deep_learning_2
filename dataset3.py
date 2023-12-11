import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import json

class TextDataset2(Dataset):
    def __init__(self, context_size=8):
        self.french_chrset = set()
        self.english_chrset = set()

        self.index = [] # A list of dictionaries, containing the row and offset of the text.
        # self.df = pd.read_csv(f'bitcoin_articles.csv')
        self.context_size = context_size
        f = open(f'french_english.json')
        self.contents = f.read()
        f.close()

        self.text_dict = json.loads(self.contents)['utterances']
        i = 0

        while True:
            if str(i) in self.text_dict:
                french = self.text_dict[str(i)]['original_text']
                english = self.text_dict[str(i)]['postprocessed_text']
            
                for j in range(0, len(french)):
                    if j + context_size + 1 > len(french):
                        break
                    if j + context_size > len(english):
                        continue
                    self.index.append({
                        "idx": i,
                        "offset": j
                    })
                for ch in french:
                    self.french_chrset.add(ch)

                for ch in english:
                    self.english_chrset.add(ch)
                i += 1
            else:
                break
        self.french_encode_dict = {}
        self.french_decode_dict = {}
        
        self.english_encode_dict = {}
        self.english_decode_dict = {}

        for i, x in enumerate(self.french_chrset):
            self.french_encode_dict[x] = i
            self.french_decode_dict[i] = x

        for i, x in enumerate(self.english_chrset):
            self.english_encode_dict[x] = i
            self.english_decode_dict[i] = x

    def __len__(self):
        return len(self.index)
    
    def encode_french(self, x):
        encoded = [self.french_encode_dict[ch] for ch in x]
        return encoded
    
    def decode_french(self, x):
        decode = [self.french_decode_dict[i] for i in x]
        return ''.join(decode)
    
    def encode_english(self, x):
        encoded = [self.english_encode_dict[ch] for ch in x]
        return encoded
    
    def decode_english(self, x):
        decode = [self.english_decode_dict[i] for i in x]
        return ''.join(decode)
    

    def __getitem__(self, idx):
        dict_index = self.index[idx]["idx"]
        dict_offset = self.index[idx]["offset"]
        # print(dict_index)
        # print(dict_offset)
        french = self.text_dict[str(dict_index)]["original_text"][dict_offset:dict_offset+self.context_size]
        english = self.text_dict[str(dict_index)]["postprocessed_text"][dict_offset:dict_offset+self.context_size]
        y = self.text_dict[str(dict_index)]["original_text"][dict_offset+1:dict_offset+self.context_size+1]
        # y = self.contents[idx+1: idx + self.context_size+1]
        assert (len(french) == len(y)), f'Len of french: {len(french)} != Len of english: {len(english)}'

        return (torch.Tensor(self.encode_french(french)).long(), torch.Tensor(self.encode_english(english)).long(), 
                torch.Tensor(self.encode_french(y)).long())
    
    def get_chr_set(self):
        return self.chrset