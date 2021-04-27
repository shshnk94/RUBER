import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    
    def __init__(self, config):
        
        super(ClassifierHead, self).__init__()
        
        self.hidden1 = nn.Linear(2 * config['embedding_dim'] + 1, config['hidden1_dim'])
        self.hidden2 = nn.Linear(config['hidden1_dim'], config['hidden2_dim'])
        self.output = nn.Linear(config['hidden2_dim'], config['num_labels'])
        self.dropout = nn.Dropout(config['drop_prob'])
        
    def forward(self, x):
        
        x = F.relu(self.dropout(self.hidden1(x)))
        x = F.relu(self.dropout(self.hidden2(x)))
        x = self.output(x)
        
        return x
        
class Unreferenced(nn.Module):

    def __init__(self, config):

        super(Unreferenced, self).__init__()
    
        self.classifier = ClassifierHead(config)
        self.matrix = nn.Linear(config['embedding_dim'], config['embedding_dim'])

    def forward(self, query, generated):
        
        quad = torch.bmm(self.matrix(query).unsqueeze(1), generated.unsqueeze(2)).squeeze(2)
        logits = self.classifier(torch.cat((query, quad, generated), dim=1))

        return logits
    
class Ruber(nn.Module):
    
    def __init__(self, config):
        
        super(Ruber, self).__init__()
        self.unreferenced = Unreferenced(config)
        
    def referenced(self, reference, generated):
        return torch.bmm(reference.unsqueeze(1), generated.unsqueeze(2)).squeeze(2)
    
    def forward(self, query, reference, generated):
        
        logits = self.unreferenced(query, generated)
        score = self.referenced(reference, generated)
        
        return logits, score
