import torch

def to_tensor(batch):
    
    query = torch.cat([x[0].unsqueeze(0) for x in batch], dim=0)
    reference = torch.cat([x[1].unsqueeze(0) for x in batch], dim=0)
    generated = torch.cat([x[2].unsqueeze(0) for x in batch], dim=0)
    
    labels = torch.tensor([x[3] for x in batch], dtype=torch.long)
    
    return query, reference, generated, labels
