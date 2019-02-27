import torch, torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NegSampleLoss(nn.Module):

	def __init__(self, delta = 0.5):
		
		super(NegSampleLoss, self).__init__()

		self.delta = delta

	def forward(self, positive_score, negative_score):

		return torch.max(torch.Tensor([0]).to(device), self.delta - positive_score + negative_score)
