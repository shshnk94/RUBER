import os

import pandas as pd
import torch
from torch.utils.data import Dataset

class TextPadSequence:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
            
        return self.tokenizer(batch, 
                              padding=True, 
                              return_token_type_ids=False, 
                              return_tensors='pt')

class TextDataset(Dataset):
    
    def __init__(self, config):
        
        def remove_nans(df):
            
            df.loc[df['text'].isnull(), 'text'] = ''            
            return df
        
        self.text = remove_nans(pd.read_csv(os.path.join(config['path'], config['mode'], 'text.csv')))
        
    def __len__(self):
        return self.text.shape[0]
        
    def __getitem__(self, index):
        return self.text.loc[index, 'text']
