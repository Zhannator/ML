import sys
import string
import math
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords

# Constants
TOTAL_CLASSES = 20
TOTAL_CLASS_ARTICLES =  20000
CLASS_ARTICLES = 1000
TOTAL_TRAINING_ARTICLES = 10000
TOTAL_CLASSIFICATION_ARTICLES = 10000
TRAINING_ARTICLES = 500

CLASS_NAMES = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", \
"misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "soc.religion.christian", \
"talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc", "alt.atheism", "sci.space", \
"sci.crypt", "sci.electronics", "sci.med"]


# Parse text into words
def parse_text(text):
	# Process text from the article
	words = text.split()
	# Remove all symbols and change to lowercase
	translation_table = str.maketrans('', '', string.punctuation)
	words = [(word.translate(translation_table)).lower() for word in words]
	# Get rid of extra words
	stop_words = set(stopwords.words('english'))
	words = [word for word in words if not word in stop_words]
	return words

def main():
	# Main starts here

	# 20 Newsgroup Classes  {"class-name" : [{"word" : num_of_occurances}, total_number_of_words]}
	c = {"comp.graphics" : [{}, 0], "comp.os.ms-windows.misc" : [{}, 0], "comp.sys.ibm.pc.hardware" : [{}, 0], "comp.sys.mac.hardware" : [{}, 0], "comp.windows.x" : [{}, 0], \
             "misc.forsale" : [{}, 0], "rec.autos" : [{}, 0], "rec.motorcycles" : [{}, 0], "rec.sport.baseball" : [{}, 0], "rec.sport.hockey" : [{}, 0], "soc.religion.christian" : [{}, 0], \
	     "talk.politics.guns" : [{}, 0], "talk.politics.mideast" : [{}, 0], "talk.politics.misc" : [{}, 0], "talk.religion.misc" : [{}, 0], "alt.atheism" : [{}, 0], "sci.space" : [{}, 0], \
	     "sci.crypt" : [{}, 0], "sci.electronics" : [{}, 0], "sci.med" : [{}, 0]}

	total_number_words = 0

	Pw = {}

	# Train set of  500 documents from every newsgroup class
	for news_class in c:
		print ("Training class: " + news_class)
		filedir = "20_newsgroups/" + news_class + "/"
		list_of_article_titles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
		# Cycle through the first 500 articles in this class
		c_dict = [{}, 0]
		for i in range(TRAINING_ARTICLES):
			# Read in text from the article
			#print (list_of_article_titles[i]+"\n")
			file = open(filedir + list_of_article_titles[i], 'rt', errors = 'ignore')
			text = file.read()
			words = parse_text(text)
			#print (words[:100])
			# Count how many times you saw word in documents of this topic
			for w in words:
				if (w != ""):
					if w in c_dict[0]:
						c_dict[0][w] += 1
					else:
						c_dict[0][w] = 1
					c_dict[1] += 1 # Update total number of words in c
		for feature in c_dict[0]:
			count = c_dict[0][feature]
			if (count >= 10):
				c[news_class][0][feature] = c_dict[0][feature]
				c[news_class][1] = c[news_class][1] + count
				total_number_words += 1
				# Keep track of unique set of words and their count
				if feature in Pw:
					Pw[feature] += count
				else:
					Pw[feature] = count

	# Calculate P(c) - constant for all classes
	Pc = TRAINING_ARTICLES/TOTAL_TRAINING_ARTICLES

	# Calculate P(w) - constant for all classes
	for word in Pw:
		Pw[word] = Pw[word] / total_number_words

	num_correct_classifications = 0

	# Classify set of 500 documents from every newsgroup class and store results in results.txt [Class Article_Number Classified_As Probability Correctness]
	f_results = open("results.txt", "a")
	for news_class in c:
		print("Classifying class: " + news_class)
		filedir = "20_newsgroups/" + news_class + "/"
		list_of_article_titles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
		#print (len(list_of_article_titles))
		# Cycle through the second half of articles in this class
		for i in range(TRAINING_ARTICLES, len(list_of_article_titles)):
			# Read in text from the article
			file = open(filedir + list_of_article_titles[i], 'rt', errors='ignore')
			text = file.read()
			file.close()
			words = parse_text(text)
			# Find all unique features
			X = [] # X contains a list of unique features and their quantities
			for w in words:
				if (w != ""):
					if w not in X:
						X.append(w)
			# Calculate P(c | w)
			Pcw = []
			X_length = len(X)
			# For every class, calculate P(w | c)
			for j in range(TOTAL_CLASSES):
				Pwc = 1
				# For every word in X, find P(w | c) and multiply them together to get total probability
				for k in X:
					# Account for words that do not occur in training and avoid Pcw being unintentionally set to 0
					w_quantity_in_c = 1
					if k in c[news_class][0]:
						w_quantity_in_c = w_quantity_in_c + c[news_class][0][k] # words w in class news_class
					total_words_in_c = c[news_class][1] + X_length
					Pwc = Pwc * (w_quantity_in_c / total_words_in_c)
				Pcw.append(Pwc) # No need to divide by Pw or multiply by Pc because it will be the same for every class
			#print (Pcw)
			Pcw_max = max(Pcw) # Maximum probability
			c_classified = CLASS_NAMES[Pcw.index(Pcw_max)]
			correctness = 0
			if (c_classified == news_class):
				correctness = 1
				num_correct_classifications += 1
			#print ("{}\n\n{}\n\n{}\n\n{}\n\n{}\n".format(news_class, list_of_article_titles[i], c_classified, Pcw_max,correctness))
			f_results.write("{} {} {} {} {}\n".format(news_class, list_of_article_titles[i], c_classified, Pcw_max,correctness))
	f_results.close()
	print ("Number of correct classifications: " + str(num_correct_classifications) + " / " + str(TOTAL_CLASSIFICATION_ARTICLES) + " = " + str(num_correct_classifications / TOTAL_CLASSIFICATION_ARTICLES))

if __name__ == "__main__":
	main()
