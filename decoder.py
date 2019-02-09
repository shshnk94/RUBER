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

	def forward(self, Y, Y_lengths, context_vector):
		
		batch_size, max_Y_length = Y.size()

		# Convert the dataset of indices to their respective word embeddings (including padding).
		# (num_sentences, max_Y_length) => (num_sentences, max_Y_length, embedding_dim)
		Y = self.embedding(Y)

		# Pack padded sequence to avoid computations on the padding
		Y_lengths, sorted_indices = -np.sort(-Y_lengths), np.argsort(-Y_lengths)
		Y = Y.index_select(0, torch.LongTensor(sorted_indices))
		Y = nn.utils.rnn.pack_padded_sequence(Y, Y_lengths, batch_first = True)
		
		# Feed forward through the GRU
		packed_output, h_N = self.GRU(Y, context_vector)
		
		# Unpack the output to feed it to further layers. Unpacked output is a vector formed by the concatenation of the
		# outputs from both the forward and backward RNNs.
		gru_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first = True)
		
		# In case we need to de-concatenate the output.
		# output = output.view(batch_size, max_Y_length, self.num_directions, -1)

				
		# Code to handle BiGRU output and then pass it to the Linear and Softmax layers 
		output = self.output(self.signal(gru_output))
		
		return output
