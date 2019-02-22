import tweepy
import sys
import os
import pdb

def initilialize(function):
	
	try:
		os.mkdir(function)
	except FileExistsError as error:
		print("Warning : Using the existing folder")
 
	context = open(function + "/context.txt", "wb", buffering = 0)
	response = open(function + "/response.txt", "wb", buffering = 0)

	consumerKey = 'tO8bYzSAuMGcmV1dyXMZKCYKK'
	consumerSecret = 'mJ1u1da7rn5cagPgJjGgZTdsri6dAhW486tSMJopy3htwqVAu2'
	accessToken = '850213589067546626-yxSG3Ex8Rz7p4EVCxIYxOjPW0JqEOc3'
	accessTokenSecret = 'Jbb7ing4fHEjJy5F8SoldyFqJPqFdUnFfIYMT0zo6Ibvm'
	
	auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
	auth.set_access_token(accessToken, accessTokenSecret)
	api = tweepy.API(auth)
	
	return api, context, response

def writeToFile(triple, context, response):
	
	for index, string in enumerate(triple):
		
		if string[0] == '@':
			triple[index] = " ".join(string.split()[1:])
		
		if index == 2:
			response.write(("<secondSpeaker> " + triple[index] + " </secondSpeaker>\n").encode('utf-8'))
		else:
			if index:
				context.write(("<secondSpeaker> " + triple[index] + " </secondSpeaker>\n").encode('utf-8'))
			else:
				context.write(("<firstSpeaker> " + triple[index] + " </firstSpeaker> ").encode('utf-8'))
	
	return			

if __name__ == "__main__":

	inputFileName = sys.argv[1]
	function = sys.argv[2]
	api, context, response = initilialize(function)

	with open(inputFileName, "r") as handle:
			
		for line in handle:
			
			triple = []

			for tweetID in line.split():

				try:
					triple.append(api.get_status(tweetID)._json['text'])
				except:
					triple.append("")

			if "" not in triple:
				
				#print(triple)
				writeToFile(triple, context, response)
