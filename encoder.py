import torch, torch.nn as nn

def create_embedding_layer(weight_matrix):

	num_embeddings, embedding_dim = weight_matrix.size()

	embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
	embedding_layer.load_state_dict({"weight" : weight_matrix})
	embedding_layer.requires_grad = False

	return embedding_layer
	

class Encoder(nn.Module):
	
	def __init__(self, input_size, hidden_size, weight_matrix, batch_size = 1, num_layers = 1, num_directions = 2):
		
		super(Encoder, self).__init__()
		
		# Initialize encoder parameters	
		self.input_size = input_size
		self.num_directions = num_directions
		self.hidden_size = hidden_size	
		self.batch_size = batch_size
		self.embedding = create_embedding_layer(weight_matrix)
		
		# Create a GRU encoder with the specified parameters
		self.GRU = nn.GRU(input_size = self.input_size,
						  hidden_size = self.hidden_size,
						  bidirectional = True if (num_directions == 2) else False)

		return

	def forward(self, sentence):
		
		# Initialize the forward and backward GRU states with a zero vector
		h0 = torch.zeros((self.num_directions, self.batch_size, self.hidden_size))

		output, h_N = self.GRU(self.embedding(sentence), h0)
		
		return output, h_N
		
		
