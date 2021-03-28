import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiGRUSentence(nn.Module):
	
    def __init__(self, input_size, hidden_size, weight_matrix):
		
	super(SentenceEmbedding, self).__init__()
		
	# Initialize encoder parameters	
	self.input_size = input_size
	self.hidden_size = hidden_size
	self.embedding = create_embedding_layer(weight_matrix)
	self.batch_first = True
		
	# Keeping this hardcoded for now	
	self.num_directions = 2
	self.batch_size = 1 # We iterate across the dataset during training
		
	# Create a GRU encoder with the specified parameters
	self.GRU = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          bidirectional = True)
	
	return
	
    def forward(self, x):
		
	# Initialize the forward and backward GRU hidden states before processing each batch
	hidden = torch.zeros((self.num_directions,  self.batch_size, self.hidden_size))

	# Convert the dataset of indices to their respective word embeddings and,
	# change (sentence_length, embedding_dim) to (sentence_length, batch_size = 1, embedding_dim).
	x = self.embedding(x)
	x = x.view(x.size()[0], 1, x.size()[1])

	# Feed forward through the GRU
	output, hidden = self.GRU(x, hidden)
	
	# Concatenate the final states from backward and forward GRUs
	hidden = hidden.squeeze()
	hidden = torch.cat((hidden[0], hidden[1]))

	return hidden

class PoolSentence(nn.Module):
    pass

#FIXME: Maybe a dicitonary here to decide the model.
def get_sentence_embedding(config):
    
    if config['model'] == 'ruber':
        return BiGRUSentence()

    else:
        return 

def get_classifier(config):

    if config['model'] == 'ruber':
        return Perceptron()

    else:
        return 

class Unreferenced(nn.Module):

    def __init__(self, config):
		
	super(Model, self).__init__()

	# FIXME : Hardcoded sizes need to be managed generalized
	self.sentence = get_sentence_embedding(config)
	self.classifier = get_classifier(config)
		
	# Parameter matrix
        self.matrix = nn.Linear(20, 20)
	#init_weights(self)

	return

    def forward(self, query, response):
				
	# Forward prop involves finding the sentence embedding of both query and response
	q = self.sentence(query)
	r = self.sentence(response)

	# Calculate the quadratic feature and append it between 'q' and 'r'	
	quad = torch.matmul(self.matrix(q), r)
	score = self.classifier(torch.cat((q, quad.view(1), r)))

	return score
