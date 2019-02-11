import torch, torch.optim as optim, torch.nn as nn
import pickle as pkl
import numpy as np

# Custom Encoder and Decoder classes for the respective RNNs
from encoder import Encoder
from decoder import Decoder

def load_data():
    
	# Read the context and responseword embeddings
	with open("tuning/context.pkl", "rb") as context_file:
		X = pkl.load(context_file)
	with open("tuning/response.pkl", "rb") as response_file:
		Y = pkl.load(response_file)
	
	# Reads the weight matrix required to create the embedding layer and also word indices.
	with open("vocab_weight_matrix.pkl", "rb") as weight_file:
		weight_matrix = pkl.load(weight_file)
	with open("word_to_index.pkl", "rb") as index_file:
		word_to_index = pkl.load(index_file)
	with open("vocabulary.pkl", "rb") as vocab_file:
		vocabulary = pkl.load(vocab_file)
 
	return X, Y, weight_matrix, word_to_index, vocabulary

def change_word_to_index(context, response, word_to_index, vocabulary):
	
	X = []
	Y = []

	for x, y in zip(context, response):
	
		# Convert all words to their integer indices	
		sentence = [word_to_index[word] for word in x]
		X.append(torch.LongTensor(sentence))
	
		# Same as above				
		sentence = [word_to_index[word] for word in y]
		# Add the indexes for <SOS> and <EOS> tokens into the response sentence
		sentence.insert(0, word_to_index["<SOS>"])
		sentence.append(word_to_index["<EOS>"])	
		Y.append(torch.LongTensor(sentence))
		
	# Return the datasets with each sentence as tensor of indices
	return X, Y

def train(context, response, encoder_optimizer, decoder_optimizer, criterion):
	
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	
	loss = 0
	
	# Creates a summary of the input context by producing a context vectors for context in X:
	encoder_outputs, summary_vector = encoder_rnn.forward(context)

	# Use the context vector and generate the response sentence.
	# Without using Teacher Forcing and using the predictions as inputs
	decoder_hidden = summary_vector

	for i, word in enumerate(response):
		
		# Break the loop whe EOS		
		if i == len(response) - 1:
			break

		decoder_output, decoder_hidden = decoder_rnn.forward(word, decoder_hidden)
		loss += criterion(decoder_output[0], torch.LongTensor([response[i+1]]))

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / (len(response) - 1)
	
def train_iters(encoder_rnn, decoder_rnn, num_epochs = 100, learning_rate = 0.01):
	
	encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr = learning_rate)

	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		
		for context, response in zip(X, Y):
			train(context, response, encoder_optimizer, decoder_optimizer, criterion) 

if __name__ == "__main__":

	"""
	Each item in the list X contains one context sentence with each word embedding of size 25.
	Hence an item has the dimension (number_of_words, embedding_dimension = 25). Similarly the response Y.
	"""
	X, Y, weight_matrix, word_to_index, vocabulary = load_data()
	X, Y = change_word_to_index(X, Y, word_to_index, vocabulary)

	encoder_rnn = Encoder(25, 10, torch.Tensor(weight_matrix))
	decoder_rnn = Decoder(25, 10, len(vocabulary),  torch.Tensor(weight_matrix))
	
	train_iters(encoder_rnn, decoder_rnn, 10)
	#output = decoder_rnn.forward(Y, Y_lengths, context_vector)
