from nltk.tokenize import TweetTokenizer
import sys
import pickle as pkl
import numpy as np

embedding_dim = 50

"""
This function loads the data (context/resposne) from the text files, tokenizes and stores it for further usage.
Also, it creates multiple indices for easier access of the words.

Parameters
__________

filename : Name of the file to be tokenized
vocabulary : List of words or vocabulary from the observed datasets.
word_to_index : Dictionary mapping the word to it's index.
index : Variable to maintain subsequent indices

Returns
_______

X : Tokenized data
index : Variable to maintain subsequent indices (updated).
"""
def load_and_tokenize(filename, vocabulary, word_to_index, index):
	
	# Using the tokenizer provided in the nltk.tokenize.casual module	
	tokenizer = TweetTokenizer()	
	X = []

	with open(filename, "rb") as handle:
		data = pkl.load(handle)
		
	for line in data:

		tokenized_line = tokenizer.tokenize(line)
		sentence = []

		for string in tokenized_line:
				
			# Update the indices when a new word is encountered.
			if string not in vocabulary:
				vocabulary.append(string)
				word_to_index[string] = index
				index += 1

			sentence.append(string)

		X.append(sentence)

	return X, index

"""
Stores the tokenized datasets for further usage.

Parameters	
__________

function : Purpose for which data the is used, either tuning or validation.
file_type : Specifies if the file is the context file or response.
data : Tokenized data.
"""
def store_data(filename, data):
	
	with open(filename, "wb") as handle:
		pkl.dump(data, handle)

	return

"""
Create a embedding matrix (weight matrix) for the vocabulary of words in the dataset.

Parameters
__________

vocabulary : Vocabulary of words from the training and testing datasets.
"""
def create_embedding_matrix(vocabulary):
	
	with open("word_to_embedding.pkl", "rb") as handle:
		word_to_embedding = pkl.load(handle)
	
	weight_matrix = []
	for word in vocabulary:
		
		# Initialize random vectors to the special tokens.		
		if (word == "<EOS>") or (word == "<SOS>") or (word == "<UNK>"):
			#weight_matrix.append(np.random.rand(embedding_dim))
			weight_matrix.append(np.zeros(embedding_dim))
		
		else:
			try:
				weight_matrix.append(word_to_embedding[word])
			except:
				# When the word is not found in the GloVe dictionary, assign the embedding of the UNK token.
				weight_matrix.append(weight_matrix[word_to_index["<UNK>"]])
	
	weight_matrix = np.array(weight_matrix)

	with open("weight_matrix.pkl", "wb") as handle:
		pkl.dump(weight_matrix, handle)

	return

vocabulary = ["<EOS>", "<SOS>", "<UNK>"]
word_to_index = {"<EOS>" : 0, "<SOS>" : 1, "<UNK>" : 2}
file_path = ["X_train.pkl", "Y_train.pkl", "X_test.pkl", "Y_test.pkl"]
	
index = len(vocabulary)
for filename in file_path:
	data, index = load_and_tokenize(filename, vocabulary, word_to_index, index)
	store_data(filename, data)

with open("vocabulary.pkl", "wb") as handle:
	pkl.dump(vocabulary, handle)
with open("word_to_index.pkl", "wb") as handle:
	pkl.dump(word_to_index, handle)

create_embedding_matrix(vocabulary)
