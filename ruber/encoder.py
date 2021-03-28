import os
import argparse
import json

import h5py
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
                            shuffle=True, 
                            collate_fn=pad_sequence)

    else:
        
        model, processor = build_model(config)
        
        pad_sequence = SpeechPadSequence(config, processor)
        loader = DataLoader(SpeechDataset(config), 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            collate_fn=pad_sequence)
    
    handle = h5py.File(os.path.join(config['embedding_path'], config['mode'], config['data_type'] + '_embeddings.h5'), 'w')
    src_dataset = handle.create_dataset('dataset',
                                        (len(loader.dataset), config['embedding_dim']),
                                        maxshape=(None,), 
                                        chunks=True,
                                        dtype=h5py.special_dtype(vlen=np.dtype('int32')))
    #embeddings = []
    model.eval()
    with torch.no_grad():
       
        for batch in loader:
            
            if config['data_type'] == 'text':
                batch = model(batch['input_ids'].to(device), 
                              batch['attention_mask'].to(device)).last_hidden_state[:, 0] #[CLS] token

            else:
                batch = model(batch).last_hidden_state.max(dim=1).values
                
            embeddings.append(batch)
    
    #embeddings = torch.cat(embeddings)
    torch.save(embeddings, os.path.join(config['embedding_path'], config['mode'], config['data_type'] + '_embeddings.pt'))
    
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encodes text/speech to their respective embeddings.')
    parser.add_argument('--config', type=str, help='path to the config json')
    args = parser.parse_args()

    #     'sr': 16000,
#config['model'] = 'facebook/wav2vec2-base-960h'

    with open(args.config, 'r') as f:
        config = json.load(f)

    encode(config)
