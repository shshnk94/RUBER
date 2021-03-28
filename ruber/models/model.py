import torch, torch.nn as nn

from sentence_embedding import SentenceEmbedding
from perceptron import Perceptron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
	
	for n, p in m.named_parameters():

		# Don't do anything special for summary and M
		if n == "summary.embedding.weight":
			continue
		
		# Initialize bias values with small constant	
		if "bias" in n:
			p.data.fill_(0.1)
		
		#FIXME: No sure if this is used for tanh as well.
		# Initialize rest of the weights with Xavier initialization
		else:
			torch.nn.init.xavier_uniform_(p.data)

class Model(nn.Module):
	
	def __init__(self, weight_matrix):
		
		super(Model, self).__init__()

		# FIXME : Hardcoded sizes need to be managed generalized
		self.summary = SentenceEmbedding(50, 10, weight_matrix)
		self.mlp = Perceptron(41).to(device)
		
		# Parameter matrix
		self.M = nn.Parameter(torch.Tensor(20, 20).to(device))
		
		init_weights(self)

		return

	def forward(self, context, response):
				
		# Forward prop involves finding the sentence embedding of both query and response
		q = self.summary(context)
		r = self.summary(response)

		# Calculate the quadratic feature and append it between 'q' and 'r'		
		quad = torch.matmul(torch.matmul(q, self.M), r)
		score = self.mlp(torch.cat((q, quad.view(1), r)))

		return score
