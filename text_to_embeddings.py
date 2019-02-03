import pickle as pkl
import numpy as np
import sys
import re

def loadData(function, fileType):
	
	X = []	
	with open(function + "/" + fileType + ".txt", "r") as inputFile:
		
		for line in inputFile:
			X.append([string for string in line.split() if string[0] != "<"])

	with open("embedding_dict.pkl", "rb") as dictFile:
		wordToVec = pkl.load(dictFile)

	return X, wordToVec

def storeData(data, fileType):
	
	with open(fileType + ".pkl", "wb") as outputFile:
		pkl.dump(data, outputFile)

	return 	

if __name__ == "__main__":
	
	function = sys.argv[1]
	fileType = sys.argv[2]

	X, wordToVec = loadData(function, fileType)
	"""
	Handle OOV words by using UNK token (and it's corresponding embedding) for such words.
	Also, the used GloVe embedddings are (25x1) vectors. Hence initialize with randn(25)
	"""
	wordToVec["UNK"] = np.random.randn(25)

	data = []
	for context in X:

		sentence = []
		for word in context:
			if re.search("[a-zA-Z]", word):
				word = re.sub("\)|\-|\)|\?|\!|\.|\,", "", word)
			try:
				sentence.append(wordToVec[word])
			except:
				sentence.append(wordToVec["UNK"])

		data.append(np.array(sentence))
	
	storeData(data, fileType)
