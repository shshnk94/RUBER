import torch, torch.optim as optim, torch.nn as nn
torch.manual_seed(1)

import pickle as pkl
import numpy as np

import random
random.seed(1)

import time
import csv

from model import Model
from neg_sample_loss import NegSampleLoss

import pdb

# Use GPU/CPU based on the availability
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
		X.append(torch.LongTensor(sentence))
	
		# Same as above				
		sentence = [word_to_index[word] for word in y]
		Y.append(torch.LongTensor(sentence))
		
	# Return the datasets with each sentence as tensor of indices
	return X, Y

def train(model, X, Y, num_epochs = 100, learning_rate = 0.1):
	
	op_file = open("stats.csv", "w")	
	model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)

	#Debug prints
	#for group in model_optimizer.param_groups:
		#print("Weight Decay ", group['weight_decay'])
		#print("Learning Rate ", group['lr'])
		#print("Momentum ", group['momentum'])
	#Ends here	

	criterion = NegSampleLoss()
	epoch_statistics = []

	for epoch in range(num_epochs):
		begin_time = time.time()
		
		max_score = float('-inf')	
		min_score = float('inf')
		sum_score = sum_loss = 0.0
		
		iter_count = 0
		
		for context, response in zip(X, Y):
			
			model_optimizer.zero_grad()

			# Forward propagation	
			positive_score = model.forward(context, response)
		

			# Calculate the score for a negative sample
			negative_response = random.sample(Y, 1)[0]
			negative_score = model.forward(context, negative_response)	
			
			loss = criterion(positive_score, negative_score) 
			
			#print(loss)

			loss.backward()		

			# Debug prints
			#for n, p in model.named_parameters():
				#print(n, p, p.grad)
			#for n, p in model.named_parameters():
				#print(n, p, p.grad)
				#print(n, torch.abs(p.grad.data).max())
				#if n == "mlp.fc2.weight":
					#print(p)
					#print(p.grad)
			#Ends here

			#pdb.set_trace()
			model_optimizer.step()
		
			# Debug	
			#for n, p in model.named_parameters():
				#if n == "mlp.fc2.weight":
					#print(p)
			# Ends here

			# Statistical book keeping
			max_score = positive_score if positive_score > max_score else max_score
			min_score = positive_score if positive_score < min_score else min_score
			sum_score += positive_score
			sum_loss += loss

			if (iter_count % 300 == 0) or (iter_count == len(X) - 1):
				line = "Iter "+ str(iter_count)+ " in epoch "+ str(epoch)+ ","+ str(float(sum_score) / (iter_count + 1))+ ","+ str(float(sum_loss) / (iter_count + 1)) + "\n"
				op_file.write(line)
			
			iter_count += 1
			
			break

		end_time = time.time()
		
		for n, p in model.summary.GRU.named_parameters():
			print(n, p, p.grad)
		#print("Epoch count ", epoch, "Loss is ", sum_loss / len(X))
		#print("end")
		#for n, p in model.named_parameters():
			#print(n, p, p.grad)
			#print(n, torch.abs(p.data).sum(), torch.abs(p.grad).sum())
		
		
		# Epoch statistics	
		print("Epoch ", epoch, " loss ", sum_loss/len(X))
		epoch_statistics.append([float(max_score), float(min_score), float(sum_score / len(X)), float(sum_loss / len(X))])

		# Store the model parameters after each epoch
		with open("model_params.pkl", "wb") as handle:
			pkl.dump([parameter for parameter in model.parameters()], handle)
		
		break

	with open("epoch_statistics.pkl", "wb") as handle:
			pkl.dump(epoch_statistics,handle)
	
	op_file.close()	
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

	# Just for debugging purpose make the dataset smaller
	X_train = X_train[0:100]
	Y_train = Y_train[0:100]

	X_train, Y_train = change_word_to_index(X_train, Y_train, word_to_index, vocabulary)
	X_test, Y_test = change_word_to_index(X_test, Y_test, word_to_index, vocabulary)

	model = Model(torch.Tensor(weight_matrix).to(device))
	
	train(model, X_train, Y_train, 100, 0.0001)
	scores = test(model, X_test, Y_test)

	with open("output.pkl", "wb") as handle:
		pkl.dump(scores, handle)
