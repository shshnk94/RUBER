import os

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpeechPadSequence:

    def __init__(self, config, processor):
        self.processor = processor
        self.sampling_rate = config['sr']

    def __call__(self, batch):
        
        batch = self.processor(batch,
                               padding=True,
                               sampling_rate=self.sampling_rate,
                               return_tensors='pt')['input_values']

        return batch

class SpeechDataset(Dataset):

    def __init__(self, config):
         
        self.audio_path, self.sampling_rate = config['audio_path'], config['sr']
        self.filenames = pd.read_csv(os.path.join(config['meta_path'], 'speech.csv'))['sent_id']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return librosa.load(os.path.join(self.audio_path, self.filenames[index] + '.wav'), sr=self.sampling_rate)[0]
