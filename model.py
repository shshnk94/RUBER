import torch, torch.nn as nn

from sentence_embedding import SentenceEmbedding
from perceptron import Perceptron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
	
	def __init__(self, weight_matrix):
		
		super(Model, self).__init__()

		# FIXME : Hardcoded sizes need to be managed generalized
		self.summary = SentenceEmbedding(50, 10, weight_matrix)
		self.mlp = Perceptron(41).to(device)
		
		# Parameter matrix
		self.M = nn.Parameter(torch.Tensor(20, 20).to(device))

		return

	def forward(self, context, response):
				
		# Forward prop involves finding the sentence embedding of both query and response
		q = self.summary(context)
		r = self.summary(response)

		# Calculate the quadratic feature and append it between 'q' and 'r'		
		quad = torch.matmul(torch.matmul(q, self.M), r)
		score = self.mlp(torch.cat((q, quad.view(1), r)))

		return score
