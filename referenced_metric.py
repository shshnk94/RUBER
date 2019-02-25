import pickle as pkl
import random
import torch

from sentence_embedding import create_embedding_layer

def generate_dataset():
	
	with open("tuning/response.pkl", "rb") as handle:
		response = pkl.load(handle)

	# Create the custom generated response, half of them right and the rest negative responses.
	generated = response[0:int(len(response)/2)]
	generated += random.sample(response, len(response) - int(len(response)/2))

	return response, generated

def load_supplement():
	
	with open("word_to_index.pkl", "rb") as handle:
		word_to_index = pkl.load(handle)

	with open("embedding_dict.pkl", "rb") as handle:
		embedding_dict = pkl.load(handle)

	with open("vocab_weight_matrix.pkl", "rb") as handle:
		weight_matrix = pkl.load(handle)

	return word_to_index, embedding_dict, weight_matrix

def change_word_to_index(word_to_index, response, generated):
	
	X = []
	Y = []

	for x, y in zip(response, generated):
	
		# Convert all words to their integer indices	
		sentence = [word_to_index[word] for word in x]
		X.append(torch.LongTensor(sentence))
	
		# Same as above				
		sentence = [word_to_index[word] for word in y]
		Y.append(torch.LongTensor(sentence))
		
	# Return the datasets with each sentence as tensor of indices
	return X, Y

def get_summary(sentence, embedding, UNK_emb):
	
	sentence = embedding(sentence)

	max_vector, arg_max = torch.max(sentence, dim = 0)
	min_vector, arg_min = torch.min(sentence, dim = 0)

	# Code to be deleted
	max_count = 0
	for word in arg_max:
		if not (0 in (sentence[word] == UNK_emb)):
			max_count += 1

	min_count = 0
	for word in arg_min:
		if not (0 in (sentence[word] == UNK_emb)):
			min_count += 1
	print(max_count,",",min_count, sep = "")
	
	#print(torch.cat((max_vector, min_vector)))
	return torch.cat((max_vector, min_vector))
	
"""
Function to be deleted.
"""
def count_unk(sentence, UNK_emb, embedding):
	
	sentence = embedding(sentence)
	count = 0
	for word in sentence:
		if not (0 in (word == UNK_emb)):
			count += 1

	return count
	
def reference_based_score(response, generated, embedding, UNK_emb):
		
		scores = []
	
		for r, g in zip(response, generated):
		
			# Code to be deleted.	
			r_count = count_unk(r, UNK_emb, embedding)
			g_count = count_unk(g, UNK_emb, embedding)
			#print(len(r),",",r_count,",",len(g),",",g_count,",", end = "", sep = "")
			# Ends here
			r = get_summary(r, embedding, UNK_emb)
			g = get_summary(g, embedding, UNK_emb)

			#Code to be deleted
			#print(float(torch.dot(r, g) / (r.norm() * g.norm())))
			# Ends here
			scores.append(float(torch.dot(r, g) / (r.norm() * g.norm())))
			
		return scores
			
if __name__ == "__main__":
	
	response, generated = generate_dataset()
	word_to_index, embedding_dict, weight_matrix = load_supplement()

	response, generated = change_word_to_index(word_to_index, response, generated)
	embedding = create_embedding_layer(torch.Tensor(weight_matrix))
	
	#print("Truth_total,Truth_UNK,Gen_total,Gen_UNK,Score")
	print("Max,Min")
	scores = reference_based_score(response, generated, embedding, torch.Tensor(weight_matrix[2]))
	
	#print(scores)	
	with open("scores.pkl", "wb") as handle:
		pkl.dump(scores, handle)
