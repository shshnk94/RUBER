import torch, torch.nn as nn
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

def word_to_index_with_padding(context, response, word_to_index, vocabulary):
	
	# Get the length of each sentence and the max among them
	X_lengths = []
	Y_lengths = []

	for x, y in zip(context, response):
		X_lengths.append(len(x))
		Y_lengths.append(len(y))

	max_X_length = max(X_lengths)
	max_Y_length = max(Y_lengths)
	X = []
	Y = []

	for x, y in zip(context, response):
	
		# Convert all words to their integer indices	
		sentence = [word_to_index[word] for word in x]
		# Add padding to make the sentence equal to the max_X_length
		padding = [word_to_index["<PAD>"] for i in range(max_X_length - len(sentence))]
		sentence += padding
		X.append(sentence)
				
		sentence = [word_to_index[word] for word in y]
		padding = [word_to_index["<PAD>"] for i in range(max_Y_length - len(sentence))]
		sentence += padding
		Y.append(sentence)
		
	# Return the datasets as tensors
	return torch.LongTensor(X), torch.LongTensor(Y), np.array(X_lengths), np.array(Y_lengths)

"""
Each item in the list X contains one context sentence with each word embedding of size 25.
Hence an item has the dimension (number_of_words, embedding_dimension = 25). Similarly the response Y.
"""
X, Y, weight_matrix, word_to_index, vocabulary = load_data()
X, Y, X_lengths,Y_lengths = word_to_index_with_padding(X, Y, word_to_index, vocabulary)

# Creates a summary of the input context by producing a context vectors for context in X:
encoder_rnn = Encoder(25, 10, torch.Tensor(weight_matrix))
context_vector = encoder_rnn.forward(X, X_lengths)

# Use the context vector and generate the response sentence.
decoder_rnn = Decoder(25, 10, len(vocabulary),  torch.Tensor(weight_matrix))
decoder_rnn.forward(Y, Y_lengths, context_vector)

