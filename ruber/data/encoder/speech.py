import os

import librosa
import numpy as np
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
         
        self.audio_path, self.sampling_rate = os.path.join(config['path'], config['mode'], 'speech'), config['sr']
        self.meta = pd.read_csv(os.path.join(config['path'], config['mode'], 'meta.csv'))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):

        if not self.meta.loc[index, 'end_time'] - self.meta.loc[index, 'start_time']:
            return np.zeros(1)

        return librosa.load(os.path.join(self.audio_path, self.meta.loc[index, 'sent_id'] + '.wav'), sr=self.sampling_rate)[0]
