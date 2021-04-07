import os
import argparse
import json
import pickle as pkl

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

def token_to_sentences(df):
    
    df = df.groupby(by='sent_id').agg({'filenum': min,
                                       'start_time': min, 
                                       'end_time': max, 
                                       'da_token': lambda x : ' '.join(list(x))}).reset_index()
    
    return df

def merge_consecutive_utterences(df):

    utt = 0
    result = []

    while utt < df.shape[0]:

        next_utt = utt + 1
        merged_utt = df.loc[utt].to_dict()
        speaker = merged_utt['sent_id'].split('_')[1]
        current = int(merged_utt['sent_id'].split('_')[2])

        while (next_utt < df.shape[0] and
               speaker in df.loc[next_utt, 'sent_id'] and
               int(df.loc[next_utt, 'sent_id'].split('_')[2]) == current + 1):

            merged_utt['da_token'] += (' ' + df.loc[next_utt, 'da_token'])
            merged_utt['end_time'] = df.loc[next_utt, 'end_time']

            current = int(df.loc[next_utt, 'sent_id'].split('_')[2])
            next_utt += 1

        result.append(merged_utt)
        utt = next_utt

    result = pd.DataFrame(result)

    return result

def order_turns(df):

    def sort(df):
    
        df['id'] = df['sent_id'].apply(lambda x: int(x.split('_')[2]))
        df = df.sort_values('id').drop('id', axis=1)
        return df

    df = df.groupby('filenum').apply(lambda df: sort(df))
    df.index = pd.RangeIndex(0, len(df.index.to_flat_index()))
    
    return df

def clean_text(df):

    df['da_token'] = df['da_token'].apply(lambda x: x.replace('uh huh', 'okay')\
                                          .replace('um ', '')\
                                          .replace('yeah', 'yes')\
                                          .replace(" '", "'")\
                                          .replace('<MISSED> ', '')\
                                          .replace(" n't", "n't")\
                                          .replace(" 'll", "'ll"))

    return df

def remove_stopwords(df, tokenizer, stopwords):

    def remove(x):

        x = tokenizer.tokenize(x)
        x = tokenizer.convert_tokens_to_string([token for token in x if token not in stopwords])
        return x

    df['da_token'] = df['da_token'].apply(lambda x: remove(x))

    return df

def prepare_dialogues(df):

    positive = []
    for q, r, g in zip(df.index, df.index[1:], df.index[2:]):
        positive.append({'query': df.loc[q, 'sent_id'],
                         'reference': df.loc[r, 'sent_id'],
                         'generated': df.loc[g, 'sent_id']})

    positive = pd.DataFrame(positive)
    positive['label'] = 1

    negative = positive.copy()
    negative['reference'] = positive['reference'].sample(frac=1)
    negative['label'] = 0

    dialogues = pd.concat((positive, negative))

    return dialogues

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data preprocessing for RUBER')

    parser.add_argument('--datapath', type=str, help='directory containing data')
    parser.add_argument('--savepath', type=str, help='directory to store the processed data')
    parser.add_argument('--mode', type=str, help='if training/validation/testing dataset')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='model type as in huggingface/transformers')
    config = vars(parser.parse_args())
    
    df = pd.read_csv(os.path.join(config['datapath'], '{}_aligned.tsv'.format(config['mode'])), sep='\t')
    df = token_to_sentences(df)
    df = merge_consecutive_utterences(df)
    df = order_turns(df)
    df = clean_text(df)

    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, max_df=1.0, min_df=0)

    count = vectorizer.fit_transform(df['da_token']).sum(axis=0).A[0]
    idx2word = {v: k for k, v in vectorizer.vocabulary_.items()}

    #remove affirmations - why not nltk stopwords?
    if config['mode'] == 'train':
        stopwords = set([w for i, w in idx2word.items() if count[i] / count.sum() > 0.01]) - {'okay', 'yes'}
        with open(os.path.join(config['savepath'], 'stopwords.pkl'), 'wb') as f:
            pkl.dump(stopwords, f)
    else:
        with open(os.path.join(config['savepath'], 'stopwords.pkl'), 'rb') as f:
            stopwords = pkl.load(f)

    df = remove_stopwords(df, tokenizer, stopwords)

    text, meta = df[['sent_id', 'da_token']], df[['sent_id', 'start_time', 'end_time']]
    text.columns = ['sent_id', 'text']
    
    dialogues = prepare_dialogues(text)

    if not os.path.exists(os.path.join(config['savepath'], config['mode'])):
        os.makedirs(os.path.join(config['savepath'], config['mode']), exist_ok=True)
    
    sentoidx = {sent_id: index for index, sent_id in enumerate(text['sent_id'])}
    with open(os.path.join(config['savepath'], config['mode'], 'sentence_to_index.json'), 'w') as f:
        json.dump(sentoidx, f)

    text.to_csv(os.path.join(config['savepath'], config['mode'], 'text.csv'), index=False)
    meta.to_csv(os.path.join(config['savepath'], config['mode'], 'meta.csv'), index=False)
    dialogues.to_csv(os.path.join(config['savepath'], config['mode'], 'dialogues.csv'), index=False)
