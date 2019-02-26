import torch, torch.optim as optim, torch.nn as nn
import pickle as pkl
import numpy as np
import random

from model import Model
from neg_sample_loss import NegSampleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    
	# Read the context and responseword embeddings
	with open("X_train.pkl", "rb") as handle:
		X_train = pkl.load(handle)
	with open("Y_train.pkl", "rb") as handle:
		Y_train = pkl.load(handle)
	with open("X_test.pkl", "rb") as handle:
		X_test = pkl.load(handle)
	with open("Y_test.pkl", "rb") as handle:
		Y_test = pkl.load(handle)
	
	# Reads the weight matrix required to create the embedding layer and also word indices.
	with open("weight_matrix.pkl", "rb") as weight_file:
		weight_matrix = pkl.load(weight_file)
	with open("word_to_index.pkl", "rb") as index_file:
		word_to_index = pkl.load(index_file)
	with open("vocabulary.pkl", "rb") as vocab_file:
		vocabulary = pkl.load(vocab_file)
 
	return X_train, Y_train, X_test, Y_test, weight_matrix, word_to_index, vocabulary

def change_word_to_index(context, response, word_to_index, vocabulary):
	
	X = []
	Y = []

	for x, y in zip(context, response):

		# Convert all words to their integer indices	
		sentence = [word_to_index[word] for word in x]
		X.append(torch.LongTensor(sentence, device = device))
	
		# Same as above				
		sentence = [word_to_index[word] for word in y]
		Y.append(torch.LongTensor(sentence, device = device))
		
	# Return the datasets with each sentence as tensor of indices
	return X, Y

def train(model, X, Y, num_epochs = 100, learning_rate = 0.01):
	
	model_optimizer = optim.SGD(model.parameters(), lr = learning_rate)

	criterion = NegSampleLoss()

	for epoch in range(num_epochs):
			
		for context, response in zip(X, Y):
			
			model_optimizer.zero_grad()
	
			# Forward propagation	
			positive_score = model.forward(context, response)

			# Calculate the score for a negative sample
			negative_response = random.sample(Y, 1)[0]
			negative_score = model.forward(context, negative_response)	
			
			loss = criterion(positive_score, negative_score) 

			loss.backward()		
			model_optimizer.step()

	return

def test(model, X, Y):
	
	scores = []

	for context, response in zip(X, Y):
		scores.append(model.forward(context, response))
	
	return np.array(scores)	

if __name__ == "__main__":

	#Each item in the list X contains one context sentence with each word embedding of size 25.
	#Hence an item has the dimension (number_of_words, embedding_dimension = 25). Similarly the response Y.
	X_train, Y_train, X_test, Y_test, weight_matrix, word_to_index, vocabulary = load_data()
	X_train, Y_train = change_word_to_index(X_train, Y_train, word_to_index, vocabulary)
	X_test, Y_test = change_word_to_index(X_test, Y_test, word_to_index, vocabulary)

	model = Model(torch.Tensor(weight_matrix, device = device))
	
	train(model, X_train, Y_train, 1)
	scores = test(model, X_test, Y_test)

	with open("output.pkl", "wb") as handle:
		pkl.dump(scores, handle)
