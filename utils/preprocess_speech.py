import os
import glob
import argparse

import pandas as pd
from sphfile import SPHFile

def get_speech_sample_names(path):
    
    speech_samples = []
    for dname in glob.glob(os.path.join(path, '*')):
        if 'swb1_d' in dname:
            for fname in glob.glob(os.path.join(dname, 'data', '*')):
                speech_samples.append(fname)
    
    return speech_samples
     
def convert_and_save(meta, speech_samples, path, mode):
    
    if not os.path.exists(os.path.join(path, mode, 'speech')):
        os.makedirs(os.path.join(path, mode, 'speech'), exist_ok=True)

    for index, row in meta.iterrows():
    
        name = 'sw0' + row['sent_id'].split('_')[0][-4:]
        fpath = [fname for fname in speech_samples if name in fname][0]
    
        sph = SPHFile(fpath)
        sph.write_wav(os.path.join(path, mode, 'speech', row['sent_id'] + '.wav'), 
                      row['start_time'], 
                      row['end_time'])
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data preprocessing for RUBER')

    parser.add_argument('--audiopath', type=str, help='directory containing .sph files')
    parser.add_argument('--savepath', type=str, help='directory to store the processed data')
    parser.add_argument('--metapath', type=str, help='directory containing the meta.csv files')
    parser.add_argument('--mode', type=str, help='if training/validation/testing dataset')
    config = vars(parser.parse_args())

    meta = pd.read_csv(os.path.join(config['metapath'], config['mode'], 'meta.csv'))

    speech_samples = get_speech_sample_names(config['audiopath'])
    convert_and_save(meta, speech_samples, config['savepath'], config['mode'])
