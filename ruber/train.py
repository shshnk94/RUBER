import json
import argparse

from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dialogue.dataset import UnimodalDataset, MultimodalDataset
from data.dialogue.collate import to_tensor
from models.ruber import Ruber

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_loader(config):
    
    if config['mode'] == 'train':
        
        if config['model_type'] == 'hybrid':
        
            train_loader = DataLoader(MultimodalDataset(config), 
                                      batch_size=config['batch_size'], 
                                      shuffle=True, 
                                      collate_fn=to_tensor)
        
            valid_loader = DataLoader(MultimodalDataset(config), 
                                      batch_size=config['batch_size'], 
                                      shuffle=False, 
                                      collate_fn=to_tensor)
        
        else:
        
            train_loader = DataLoader(UnimodalDataset(config), 
                                      batch_size=config['batch_size'], 
                                      shuffle=True, 
                                      collate_fn=to_tensor)
        
            valid_loader = DataLoader(UnimodalDataset(config), 
                                      batch_size=config['batch_size'], 
                                      shuffle=False, 
                                      collate_fn=to_tensor)
        
        return train_loader, valid_loader

    else:

        test_loader = DataLoader(UnimodalDataset(config), 
                                  batch_size=config['batch_size'], 
                                  shuffle=False, 
                                 collate_fn=to_tensor)

        return test_loader

def train(config, checkpoint_dir='./'):
    
    model = Ruber(config).to(device)
    train_loader, valid_loader = get_loader(config)
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.unreferenced.parameters(), 
                                  lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
    
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

        avg_train_loss = total_loss / len(train_loader)            
        avg_valid_loss, score = test(model, valid_loader)
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))

        tune.report(f1score=score)

    return 
        
def test(model, loader):
    
    model.eval()

    eval_score = 0
    nb_eval_steps = 0
    results, targets = [], []

    for query, reference, generated, labels in loader:
        
        query, reference, generated, labels = (query.to(device), 
                                               reference.to(device), 
                                               generated.to(device),
                                               labels.to(device))
        
        with torch.no_grad():        
            logits, score = model(query, reference, generated)

        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        
        results.append(logits)
        targets.append(labels)
        
        tmp_eval_score = f1_score(labels, logits.argmax(axis=1), average='weighted')
        eval_score += tmp_eval_score
        eval_loss += loss

        nb_eval_steps += 1

    print('MCC: ', matthews_corrcoef(np.concatenate([x.tolist() for x in targets]), 
                                     np.concatenate(results).argmax(axis=1)))
    
    print(classification_report(np.concatenate([x.tolist() for x in targets]), 
                                np.concatenate(results).argmax(axis=1)))
    
    avg_score = eval_score/nb_eval_steps
    return avg_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains the RUBER model.')
    parser.add_argument('--config', type=str, help='path to the config json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config)
