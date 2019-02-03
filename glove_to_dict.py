import numpy as np
import pickle as pkl

if __name__ == "__main__":
		
	#words = []
	#index = 0
	#word_to_index = {}
	word_to_embedding = {}

	with open("glove.twitter.27B.25d.txt", "r") as glove_file:
	
		for line in glove_file:
			
			word_and_vector = line.split()
		
			# Index a word	
			#words.append(word_and_vector[0])	
			# Word to index map
			#word_to_index[word_and_vector[0]] = index
			# Word to vector map
			word_to_embedding[word_and_vector[0]] = np.array(word_and_vector[1:], dtype = float)
			
			#index += 1	

	# Save all these indexes in the disk for later use	
	"""
	with open("words.pkl", "wb") as word_file:
		pkl.dump(words, word_file)
	with open("word_to_index.pkl", "wb") as word_to_index_file:
		pkl.dump(word_to_index, word_to_index_file)
	"""
	with open("word_to_embedding.pkl", "wb") as word_to_embedding_file:
		pkl.dump(word_to_embedding, word_to_embedding_file)
