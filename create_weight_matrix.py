from nltk.tokenize import TweetTokenizer
import sys
import pickle as pkl
import numpy as np

def load_and_tokenize(function, file_type, vocabulary):
	
	tokenizer = TweetTokenizer()	
	X = []

	with open(function + "/" + file_type + ".txt", "r") as input_file:
		
		for line in input_file:

			tokenized_line = tokenizer.tokenize(line)
			context_sentence = []

			for string in tokenized_line:
				
				if (string[0] != "<") and (string[len(string) - 1] != ">"):
					if string not in vocabulary:
						vocabulary.append(string)
					context_sentence.append(string)

			X.append(context_sentence)

	return X
	
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
	
	vocabulary = []
	word_to_index = {}
	file_path = [["tuning", "context"],
				 ["tuning", "response"],
				 ["validation", "context"],
				 ["validation", "response"]]

	for function, file_type in file_path:
		data = load_and_tokenize(function, file_type, vocabulary)
		store_data(function, file_type, data)

	with open("vocabulary.pkl", "wb") as vocab_file:
		pkl.dump(vocabulary, vocab_file)

	create_embedding_matrix(vocabulary)	
