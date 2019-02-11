import torch, torch.nn as nn
import numpy as np

def create_embedding_layer(weight_matrix):
	
	# num_embeddings is the total words in the vocabulary of the dataset and embedding_dim is the dimension
	# of the embedding (GloVe vector imported in this case).
	num_embeddings, embedding_dim = weight_matrix.size()

	embedding_layer = nn.Embedding(num_embeddings = num_embeddings, 
								   embedding_dim = embedding_dim,
								   padding_idx = 0) # vocabulary[0] = "<PAD>" in our case.
	embedding_layer.load_state_dict({"weight" : weight_matrix})
	embedding_layer.requires_grad = False

	return embedding_layer
	

class Encoder(nn.Module):
	
	def __init__(self, input_size, hidden_size, weight_matrix):
		
		super(Encoder, self).__init__()
		
		# Initialize encoder parameters	
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.embedding = create_embedding_layer(weight_matrix)
		self.batch_first = True
		
		# Keeping this hardcoded for now	
		self.num_directions = 2
		self.batch_size = 1 # We iterate across the dataset since the sentence lengths are different.
		
		# Create a GRU encoder with the specified parameters
		self.GRU = nn.GRU(input_size = self.input_size,
						  hidden_size = self.hidden_size,
						  bidirectional = True if (self.num_directions == 2) else False)

		return
	
	def forward(self, X):

		# Initialize the forward and backward GRU hidden states before processing each batch
		h0 = torch.zeros((self.num_directions,  self.batch_size, self.hidden_size))

		# Convert the dataset of indices to their respective word embeddings and,
		# change (sentence_length, embedding_dim) to (sentence_length, batch_size = 1, embedding_dim).
		X = self.embedding(X)
		X = X.view(X.size()[0], 1, X.size()[1])

		# Feed forward through the GRU
		output, h_N = self.GRU(X, h0)

		return output, h_N
