import os
import json

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class UnimodalDataset(Dataset):
    
    def __init__(self, config):
        
        path = os.path.join(config['path'], config['mode'])
        
        self.dialogues = pd.read_csv(os.path.join(path, 'dialogues.csv'))
        self.index = json.load(open(os.path.join(path, 'sentence_to_index.json'), 'r'))
        self.embeddings = h5py.File(os.path.join(path, '{}_embeddings.h5'.format(config['model_type'])), 'r')
        
    def __len__(self):
        return self.dialogues.shape[0]
    
    def __getitem(self, index):
        
        query, reference, generated = self.dialogues.loc[index, ['query', 'reference', 'generated']]
        query, reference, generated = (self.embeddings.get('dataset')[self.index[query]], 
                                       self.embeddings.get('dataset')[self.index[reference]], 
                                       self.embeddings.get('dataset')[self.index[generated]])
        
        return query, reference, generated, self.dialogues.loc[index, 'label']
    
class MultimodalDataset(Dataset):
    
    def __init__(self, config):
        
        path = os.path.join(config['path'], config['mode'])
        
        self.dialogues = pd.read_csv(os.path.join(path, 'dialogues.csv'))
        self.index = json.load(open(os.path.join(path, 'sentence_to_index.json'), 'r'))
            
        self.text = h5py.File(os.path.join(path, 'text_embeddings.h5'), 'r')            
        self.speech = h5py.File(os.path.join(path, 'speech_embeddings.h5'), 'r')            

    def __len__(self):
        return self.dialogues.shape[0]
        
    def __getitem__(self, index):
        
        query, reference, generated = self.dialogues.loc[index, ['query', 'reference', 'generated']]
        
        query_speech, reference_speech, generated_speech = (self.speech.get('dataset')[self.index[query]], 
                                                            self.speech.get('dataset')[self.index[reference]], 
                                                            self.speech.get('dataset')[self.index[generated]])
            
        query_text, reference_text, generated_text = (self.text.get('dataset')[self.index[query]], 
                                                      self.text.get('dataset')[self.index[reference]], 
                                                      self.text.get('dataset')[self.index[generated]])

        query, reference, generated = (torch.from_numpy(np.concatenate([query_text, query_speech])), 
                                       torch.from_numpy(np.concatenate([reference_text, reference_speech])),
                                       torch.from_numpy(np.concatenate([generated_text, generated_speech])))
            
        return query, reference, generated, self.dialogues.loc[index, 'label']
