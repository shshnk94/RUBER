import pickle as pkl
import os
from gensim.models import KeyedVectors
import random
import torch

from sentence_embedding import create_embedding_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_dataset(datapath, half_sample_length):
	
    with open(os.path.join(datapath, 'Y_train.pkl'), "rb") as handle:
	response = pkl.load(handle)

    # Generate half_sample_length samples where generated and truth are definitely same.
    indices = random.sample(range(len(response)), half_sample_length)
    truth = [response[i] for i in indices]
    generated = [response[i] for i in indices]
	
    # Generate another half_sample_length samples where most of the pairs are different
    truth += [response[i] for i in random.sample(range(len(response)), half_sample_length)]
    generated += [response[i] for i in random.sample(range(len(response)), half_sample_length)]

    return truth, generated

class ReferenceMetric:

    def __init__(datapath, half_sample_length, vocab_size, embedding_dim):

        self.datapath = datapath
        self.half_sample_length = half_sample_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)

    def load_supplement(datapath):
	
        embeddings = KeyedVectors.load(os.path.join(datapath, 'embeddings.kv'))
        return embeddings

    def get_summary(sentence, embeddings):
	
        sentence = torch.Tensor([embeddings[word] for word in sentence])
	
        max_vector, arg_max = torch.max(sentence, dim = 0)
        min_vector, arg_min = torch.min(sentence, dim = 0)
	
        return torch.cat((max_vector, min_vector))
	
    def reference_based_score(response, generated):
		
	scores = []
	
	for r, g in zip(response, generated):
		
	    r, g = get_summary(r), get_summary(g)
	    scores.append(float(torch.dot(r, g) / (r.norm() * g.norm())))
			
	return scores
			
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates reference metric')

    parser.add_argument('--datapath', type=str, help='directory containing data')
    parser.add_argument('--savepath', type=str, help='directory containing output scores')
    parser.add_argument('--halflen', type=int, help='maximum length of every caption')
    parser.add_argument('--vocab_size', type=int, help='size of the vocabulary')
    parser.add_argument('--embedding_dim', type=int, help='dimension of the embeddings')
    
    args = parser.parse_args()
	
    response, generated = generate_dataset(args.datapath, args.halflen)

    embedding_dict = load_supplement()
	
    print('Truth_total,Gen_total,Score')
    scores = reference_based_score(response, generated, embedding_dict)

    with open(os.path.join(args.savepath, 'scores.pkl'), 'wb') as handle:
	pkl.dump(scores, handle)
