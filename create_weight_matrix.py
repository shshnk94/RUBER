from nltk.tokenize import TweetTokenizer
import sys
import pickle as pkl
import numpy as np

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

def create_embedding_matrix(vocabulary):
	
	with open("word_to_embedding.pkl", "rb") as embed_file:
		word_to_embedding = pkl.load(embed_file)
	
	weight_matrix = []
	for word in vocabulary:
		try:
			weight_matrix.append(word_to_embedding[word])
		except:
			weight_matrix.append(np.random.rand(5))
	
	weight_matrix = np.array(weight_matrix)

	with open("vocab_weight_matrix.pkl", "wb") as vocab_file:
		pkl.dump(weight_matrix, vocab_file)

	return

if __name__ == "__main__":
	
	vocabulary = ["<PAD>"]
	word_to_index = {"<PAD>" : 0}
	file_path = [["tuning", "context"],
				 ["tuning", "response"],
				 ["validation", "context"],
				 ["validation", "response"]]
	
	index = 1
	for function, file_type in file_path:
		data, index = load_and_tokenize(function, file_type, vocabulary, word_to_index, index)
		store_data(function, file_type, data)

	with open("vocabulary.pkl", "wb") as vocab_file:
		pkl.dump(vocabulary, vocab_file)
	with open("word_to_index.pkl", "wb") as index_file:
		pkl.dump(word_to_index, index_file)

	create_embedding_matrix(vocabulary)	
