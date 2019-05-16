import numpy as np
import pickle as pkl
import sys

if __name__ == "__main__":
	
	# Provides option to read different filenames corresponding to different dimensional glove embeddings.
	filename = sys.argv[1]

	word_to_embedding = {}

	with open(filename, "r") as glove_file:
	
		for line in glove_file:
			
			word_and_vector = line.split()
			word_to_embedding[word_and_vector[0]] = np.array(word_and_vector[1:], dtype = float)
			
	with open("word_to_embedding.pkl", "wb") as word_to_embedding_file:
		pkl.dump(word_to_embedding, word_to_embedding_file)
