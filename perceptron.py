import torch, torch.nn as nn

class Perceptron(nn.Module):

	def __init__(self, input_size):

		super(Perceptron, self).__init__()
		
		# FIXME:Hardcoded for now	
		self.hidden_size = 10
		
		# As per the specifications
		self.output_size = 1

		self.fc1 = nn.Linear(input_size, self.hidden_size)
		self.fc2 = nn.Linear(self.hidden_size, self.output_size)

		self.dropout = nn.Dropout(0.5)

		return
		
		
	def forward(self, x):
		
		x = self.dropout(torch.tanh(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))

		return

