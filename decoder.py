import torch, torch.nn as nn
import numpy as np

"""
Same as the create_embedding_layer in the Encoder class.
"""
def create_embedding_layer(weight_matrix):
	
	num_embeddings, embedding_dim = weight_matrix.size()

	embedding_layer = nn.Embedding(num_embeddings = num_embeddings, 
								   embedding_dim = embedding_dim,
								   padding_idx = 0)
	embedding_layer.load_state_dict({"weight" : weight_matrix})
	embedding_layer.requires_grad = False

	return embedding_layer

class Decoder(nn.Module):
	
	def __init__(self, input_size, hidden_size, output_size, weight_matrix):
		
		super(Decoder, self).__init__()
		
		# Initialize encoder parameters	
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.embedding = create_embedding_layer(weight_matrix)
		
		# Keeping this hardcoded for now	
		self.num_directions = 2
		
		# Create a GRU encoder with the specified parameters
		self.GRU = nn.GRU(input_size = self.input_size,
						  hidden_size = self.hidden_size,
						  bidirectional = True if (self.num_directions == 2) else False)
		
		# Calculate the signal from the hidden layer to output layer and then pass through softmax.
		self.signal = nn.Linear(hidden_size * self.num_directions, output_size)
		self.output = nn.Softmax(dim = 2)

		return

	def forward(self, Y, hidden):
		
		# Convert the dataset of indices to their respective word embeddings and change the dimension as in Encoder
		# Since we process one word at a time the dimension should be (sentence_length = 1, batch_size = 1, embedding_dim)
		Y = self.embedding(Y)
		Y = Y.view(1, 1 , Y.size()[0])

		# Feed forward through the GRU
		gru_output, next_hidden = self.GRU(Y, hidden)
		
		# Code to handle BiGRU output and then pass it to the Linear and Softmax layers 
		output = self.output(self.signal(gru_output))
		
		return output, next_hidden
