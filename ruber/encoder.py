import os
import argparse
import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from data.text import TextPadSequence, TextDataset
from data.speech import SpeechPadSequence, SpeechDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(config):
    
    if config['data_type'] == 'text':
        
        tokenizer = AutoTokenizer.from_pretrained(config['model'])
        model = AutoModel.from_pretrained(config['model']).to(device)
        
        return model, tokenizer
    
    else:
        
        processor = Wav2Vec2Processor.from_pretrained(config['model'])
        model = Wav2Vec2Model.from_pretrained(config['model']).to(device)

        return model, processor

def encode(config):
    
    if config['data_type'] == 'text':
        
        model, tokenizer = build_model(config)
        
        pad_sequence = TextPadSequence(tokenizer)
        loader = DataLoader(TextDataset(config), 
                            batch_size=config['batch_size'], 
                            collate_fn=pad_sequence)

    else:
        
        model, processor = build_model(config)
        
        pad_sequence = SpeechPadSequence(config, processor)
        loader = DataLoader(SpeechDataset(config), 
                            batch_size=config['batch_size'], 
                            collate_fn=pad_sequence)
    
    handle = h5py.File(os.path.join(config['embedding_path'], config['mode'], config['data_type'] + '_embeddings.h5'), 'w')
    dataset = handle.create_dataset('dataset',
                                    (len(loader.dataset), config['embedding_dim']),
                                    chunks=(config['batch_size'], config['embedding_dim']),
                                    dtype=np.dtype('float32'))

    model.eval()
    with torch.no_grad():
       
        for batch_no, batch in enumerate(loader):
            
            if config['data_type'] == 'text':
                output = model(batch['input_ids'].to(device), 
                              batch['attention_mask'].to(device)).last_hidden_state[:, 0] #[CLS] token

            else:
                output = model(batch.to(device)).last_hidden_state.max(dim=1).values
                
            index = batch_no * config['batch_size']
            dataset[index: index + min(config['batch_size'], output.shape[0]), :] = output.detach().cpu().numpy()
            del output

    handle.close()
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encodes text/speech to their respective embeddings.')
    parser.add_argument('--config', type=str, help='path to the config json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    encode(config)
