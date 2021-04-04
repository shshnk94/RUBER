import os
import json

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

class UnimodalDataset(Dataset):
    
    def __init__(self, config):
        
        path = os.path.join(config['path'], config['mode'])
        
        self.dialogues = pd.read_csv(os.path.join(path, 'dialogues.csv'))
        self.index = json.load(open(os.path.join(path, 'sentence_to_index.json', 'r'))
        self.embeddings = h5py.File(os.path.join(path, '{}_embeddings.h5'.format(config['model_type'])), 'r')
        
    def __len__(self):
        return self.dialogues.shape[0]
    
    def __getitem(self, index):
        
        query, reference, generated = self.dialogues.loc[index, ['query', 'reference', 'generated']]
        query, reference, generated = (self.embeddings[self.index[query]], 
                                       self.embeddings[self.index[reference]], 
                                       self.embeddings[self.index[generated]])
        
        return query, reference, generated, self.dialogues.loc[index, 'unreferenced_label']
    
class DialogueDataset(Dataset):
    
    def __init__(self, config):
        
        path = os.path.join(config['path'], config['mode'])
        
        self.dialogues = pd.read_csv(os.path.join(path, 'dialogues.csv'))
        self.index = json.load(open(os.path.join(path, 'sentence_to_index.json', 'r'))
            
        self.text = h5py.File(os.path.join(path, 'text_embeddings.pt'), 'r')            
        self.speech = h5py.File(os.path.join(path, 'speech_embeddings.pt'), 'r')            

    def __len__(self):
        return self.dialogues.shape[0]
        
    def __getitem__(self, index):
        
        query, reference, generated = self.meta.loc[index, ['query', 'reference', 'generated']]
        
        query_speech, reference_speech, generated_speech = (self.speech[self.index[query]], 
                                                            self.speech[self.index[reference]], 
                                                            self.speech[self.index[generated]])
            
        query_text, reference_text, generated_text = (self.text[self.index[query]], 
                                                      self.text[self.index[reference]], 
                                                      self.text[self.index[generated]])

        query, reference, generated = (torch.cat([query_text, query_speech]), 
                                       torch.cat([reference_text, reference_speech]),
                                       torch.cat([generated_text, generated_speech]))
            
        return query, reference, generated, self.meta.loc[index, 'unreferenced_label']
