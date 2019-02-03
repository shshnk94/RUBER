import torch, torch.nn as nn

class Dncoder(nn.Module):
	
	def __init__(self, input_size, hidden_size, output_size, batch_size = 1, num_layers = 1, num_directions = 2):
		
		super(Dncoder, self).__init__()
		
		# Initialize encoder parameters	
		self.input_size = input_size
		self.num_directions = num_directions
		self.hidden_size = hidden_size	
		self.batch_size = batch_size
		
		# Create a GRU encoder with the specified parameters
		self.GRU = nn.GRU(input_size = self.input_size,
						  hidden_size = self.hidden_size,
						  bidirectional = True if (num_directions == 2) else False)
		
		# Output Layer
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax(output_size)

		return

	def forward(self, sentence, h0):
		
		"""	
		In case of the decoder RNN, the input sentence is same as the expected output, but after adding the
		<START> and <END> tokens to assist Teacher Forcing training.
		"""
		output, h_N = self.GRU(sentence, h0)
		
		return output, h_N
		
		
