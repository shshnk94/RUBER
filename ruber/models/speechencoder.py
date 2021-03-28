import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from ..data.audio import PadSequence, AudioDataset

class SpeechEncoder:
    
    def __init__(self, config):
        
        self.config = config
        self.processor, self.model = self.build_model()
        
    def test(self):
        
        embeddings = []
        pad_sequence = PadSequence(self.config, self.processor)
        loader = DataLoader(AudioDataset(self.config, self.processor), 
                    batch_size=self.config.batch_size, 
                    shuffle=True, 
                    collate_fn=pad_sequence)
        
        self.model.eval()
        with torch.no_grad():
            
            for batch in loader:
                embeddings.append(self.model(batch).last_hidden_state)
                
        return torch.cat(embeddings)
