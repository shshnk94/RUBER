import torch, torch.nn as nn
import pickle as pkl

# Custom Encoder and Decoder classes for the respective RNNs
from encoder import Encoder

def load_data():
    
    # Read the context and responseword embeddings
    with open("context.pkl", "rb") as context_file:
        X = pkl.load(context_file)
    with open("response.pkl", "rb") as response_file:
        Y = pkl.load(response_file)
        
    return X, Y

"""
Each item in the list X contains one context sentence with each word embedding of size 25.
Hence an item has the dimension (number_of_words, embedding_dimension = 25).
Similarly the response Y as well.
"""
X, Y = load_data()

encoder_rnn = Encoder(input_size=25, hidden_size=10)

# Creates a summary of the input context by producing a context vectors
for context in X:
	output, h_N = encoder_rnn.forward(torch.Tensor(context).view(context.shape[0], 1, context.shape[1]))

# Use the context vector and generate the response sentence.
for response in Y:
	output

