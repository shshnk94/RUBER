import pickle as pkl
from gensim.models import KeyedVectors
import random
import torch

from sentence_embedding import create_embedding_layer

def generate_dataset(half_sample_length):
	
	with open("Y_train.pkl", "rb") as handle:
		response = pkl.load(handle)

	# Generate half_sample_length samples where generated and truth are definitely same.
	indices = random.sample(range(len(response)), half_sample_length)
	truth = [response[i] for i in indices]
	generated = [response[i] for i in indices]
	
	# Generate another half_sample_length samples where most of the pairs are different
	truth += [response[i] for i in random.sample(range(len(response)), half_sample_length)]
	generated += [response[i] for i in random.sample(range(len(response)), half_sample_length)]

	return truth, generated

def load_supplement():
	
	embedding_dict = KeyedVectors.load('embeddings.kv')

	return embedding_dict

def get_summary(sentence, embedding_dict):
	
	sentence = torch.Tensor([embedding_dict[word] for word in sentence])
	
	max_vector, arg_max = torch.max(sentence, dim = 0)
	min_vector, arg_min = torch.min(sentence, dim = 0)

	return torch.cat((max_vector, min_vector))
	
def reference_based_score(response, generated, embedding_dict):
		
		scores = []
	
		for r, g in zip(response, generated):
		
			r = get_summary(r, embedding_dict)
			g = get_summary(g, embedding_dict)

			scores.append(float(torch.dot(r, g) / (r.norm() * g.norm())))
			
		return scores
			
if __name__ == "__main__":
	
	response, generated = generate_dataset(1000)

	embedding_dict = load_supplement()
	
	scores = reference_based_score(response, generated, embedding_dict)

	with open("scores.pkl", "wb") as handle:
		pkl.dump(scores, handle)
