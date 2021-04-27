import os
import json
import argparse

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from data.dialogue.dataset import UnimodalDataset, MultimodalDataset
from data.dialogue.collate import to_tensor
from models.ruber import Ruber

torch.manual_seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_loader(config):
    
    if config['mode'] == 'train':
        
        if config['model_type'] == 'hybrid':
        
            train_loader = DataLoader(MultimodalDataset(config['path'], 'train'), 
                                      batch_size=config['batch_size'], 
                                      shuffle=True, 
                                      collate_fn=to_tensor)
        
            valid_loader = DataLoader(MultimodalDataset(config['path'], 'dev'), 
                                      batch_size=config['batch_size'], 
                                      shuffle=False, 
                                      collate_fn=to_tensor)
        
        else:
        
            train_loader = DataLoader(UnimodalDataset(config['path'], 'train', config['model_type']), 
                                      batch_size=config['batch_size'], 
                                      shuffle=True, 
                                      collate_fn=to_tensor)
        
            valid_loader = DataLoader(UnimodalDataset(config['path'], 'dev', config['model_type']), 
                                      batch_size=config['batch_size'], 
                                      shuffle=False, 
                                      collate_fn=to_tensor)
        
        return train_loader, valid_loader

    else:

        if config['model_type'] == 'hybrid':
        
            test_loader = DataLoader(MultimodalDataset(config['path'], 'test'), 
                                     batch_size=config['batch_size'], 
                                     shuffle=False, 
                                     collate_fn=to_tensor)
        
        else:

            test_loader = DataLoader(UnimodalDataset(config['path'], 'test', config['model_type']), 
                                     batch_size=config['batch_size'], 
                                     shuffle=False, 
                                     collate_fn=to_tensor)

        return test_loader

def build_model(config):

    model = Ruber(config).to(device)

    #Init params using Xavier
    for name, params in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(params)
        else: #Bias initialized to zero
            nn.init.zeros_(params)
    
    return model

def train(config, checkpoint_dir='./'):
    
    model = build_model(config)
    train_loader, valid_loader = get_loader(config)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'], last_epoch=-1)
    
    tloss = open('tloss.csv', 'w') 

    for epoch in range(config['epochs']):

        model.train()

        total_loss = 0
        for query, reference, generated, labels in train_loader:

            model.zero_grad()        
            query, reference, generated, labels = (query.to(device), 
                                                   reference.to(device), 
                                                   generated.to(device),
                                                   labels.to(device))
            
            logits, score = model(query, reference, generated)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)            
        tloss.write(str(avg_train_loss) + '\n')
        score = test(model, valid_loader, criterion)
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))

        tune.report(f1score=score)
    tloss.close()
    return 
        
def test(model, loader, criterion):
    
    model.eval()
    results, targets = [], []
    vloss = open('vloss.csv', 'a') 
    loss = []
    for query, reference, generated, labels in loader:
        
        query, reference, generated, labels = (query.to(device), 
                                               reference.to(device), 
                                               generated.to(device),
                                               labels.to(device))
        with torch.no_grad():        
            logits, score = model(query, reference, generated)
        
        loss.append(criterion(logits, labels).item())
        logits = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        
        results.append(logits)
        targets.append(labels)

    targets, results = np.concatenate([x.tolist() for x in targets]), np.concatenate(results)
    mcc = matthews_corrcoef(targets, results)
    f1score = f1_score(labels, logits, average='weighted')
    acc = accuracy_score(labels, logits)
    vloss.write(str(np.mean(loss)) + '\n')
   
    print('MCC: ', mcc)
    print('F1 Score: ', f1score)
    print(classification_report(targets, results))
    vloss.close() 
    return f1score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains the RUBER model.')
    parser.add_argument('--config', type=str, default=None, help='path to the config json')
    parser.add_argument('--mode', type=str, default=None, help='decides between hp tuning/training/test')
    args = parser.parse_args()
    
    if args.config is not None:

        with open(args.config, 'r') as f:
            config = json.load(f)
            config['mode'] = args.mode
        
        if args.mode == 'train':
            
            scheduler = ASHAScheduler(time_attr='epochs',
                          metric='f1score',
                          mode='max',
                          max_t=50,
                          grace_period=5,
                          reduction_factor=2,
                          brackets=1)

            analysis = tune.run(train,
                            keep_checkpoints_num=1,
                            checkpoint_score_attr='f1score',
                            stop={"training_iteration": config['epochs']},
                            local_dir=config['output'],
                            resources_per_trial={"cpu": 2, "gpu": 0.1},
                            num_samples=1,
                            scheduler=scheduler,
                            config=config)

        else:

            criterion = nn.CrossEntropyLoss(reduction='mean')
            loader = get_loader(config)
            model = build_model(config)

            model_state, optimizer_state = torch.load(os.path.join(config['output'], 'checkpoint'))
            model.load_state_dict(model_state)

            test(model, loader, criterion)

            
            




