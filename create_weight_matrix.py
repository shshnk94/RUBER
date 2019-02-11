from nltk.tokenize import TweetTokenizer
import sys
import pickle as pkl
import numpy as np

embedding_dim = 25

"""
This function loads the data (context/resposne) from the text files, tokenizes and stores it for further usage.
Also, it creates multiple indices for easier access of the words.

Parameters
__________

function : Purpose for which data the is used, either tuning or validation.
file_type : Specifies if the file is the context file or response.
vocabulary : List of words or vocabulary from the observed datasets.
word_to_index : Dictionary mapping the word to it's index.
index : Variable to maintain subsequent indices

Returns
_______

X : Tokenized data
index : Variable to maintain subsequent indices (updated).
"""
def load_and_tokenize(function, file_type, vocabulary, word_to_index, index):
	
	# Using the tokenizer provided in the nltk.tokenize.casual module	
	tokenizer = TweetTokenizer()	
	X = []

	with open(function + "/" + file_type + ".txt", "r") as input_file:
		
		for line in input_file:

			tokenized_line = tokenizer.tokenize(line)
			context_sentence = []

			for string in tokenized_line:
				
				# Current dataset contains place holder for the type of speaker which is in XML format. Avoid them.	
				if (string[0] != "<") and (string[len(string) - 1] != ">"):
					
					# Update the indices when a new word is encountered.
					if string not in vocabulary:
						vocabulary.append(string)
						word_to_index[string] = index
						index += 1

					context_sentence.append(string)

			X.append(context_sentence)

	return X, index

"""
Stores the tokenized datasets for further usage.

Parameters	
__________

function : Purpose for which data the is used, either tuning or validation.
file_type : Specifies if the file is the context file or response.
data : Tokenized data.
"""
def store_data(function, file_type, data):
	
	with open(function + "/" + file_type + ".pkl", "wb") as output_file:
		pkl.dump(data, output_file)

	return

"""
Create a embedding matrix (weight matrix) for the vocabulary of words in the dataset.

Parameters
__________

vocabulary : Vocabulary of words from the training and testing datasets.
"""
def create_embedding_matrix(vocabulary):
	
	with open("word_to_embedding.pkl", "rb") as embed_file:
		word_to_embedding = pkl.load(embed_file)
	
	weight_matrix = []
	for word in vocabulary:
		
		# Initialize random vectors to the special tokens.		
		if (word == "<EOS>") or (word == "<SOS>") or (word == "<UNK>"):
			weight_matrix.append(np.random.rand(embedding_dim))
		
		else:
			try:
				weight_matrix.append(word_to_embedding[word])
			except:
				# When the word is not found in the GloVe dictionary, assign the embedding of the UNK token.
				weight_matrix.append(weight_matrix[word_to_index["<UNK>"]])
	
	weight_matrix = np.array(weight_matrix)

	with open("vocab_weight_matrix.pkl", "wb") as vocab_file:
		pkl.dump(weight_matrix, vocab_file)

	return

vocabulary = ["<EOS>", "<SOS>", "<UNK>"]
word_to_index = {"<EOS>" : 0, "<SOS>" : 1, "<UNK>" : 2}
file_path = [["tuning", "context"],
			["tuning", "response"],
			["validation", "context"],
			["validation", "response"]]
	
index = len(vocabulary)
for function, file_type in file_path:
	data, index = load_and_tokenize(function, file_type, vocabulary, word_to_index, index)
	store_data(function, file_type, data)

with open("vocabulary.pkl", "wb") as vocab_file:
	pkl.dump(vocabulary, vocab_file)
with open("word_to_index.pkl", "wb") as index_file:
	pkl.dump(word_to_index, index_file)

create_embedding_matrix(vocabulary)
