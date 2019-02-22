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

def get_summary(sentence, embedding):
		
	sentence = embedding(sentence)

	max_vector, _ = torch.max(sentence, dim = 0)
	min_vector, _ = torch.min(sentence, dim = 0)
	
	return torch.cat((max_vector, min_vector))
	
	
def reference_based_score(response, generated, embedding):
		
		scores = []
	
		for r, g in zip(response, generated):
			
			r = get_summary(r, embedding)
			g = get_summary(g, embedding)
			
			scores.append(float(torch.dot(r, g) / (r.norm() * g.norm())))

		return scores
			
if __name__ == "__main__":
	
	response, generated = generate_dataset()
	word_to_index, embedding_dict, weight_matrix = load_supplement()

	response, generated = change_word_to_index(word_to_index, response, generated)
	embedding = create_embedding_layer(torch.Tensor(weight_matrix))

	scores = reference_based_score(response, generated, embedding)
	print(scores)
