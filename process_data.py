import pandas as pd 
import fasttext as fst
from create_csv import create_init_df, create_init_df_sentence
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import string
stemmer = LancasterStemmer()
from keyword_extraction import *



def create_df(chemin, sentence):
	chemin = "df_brown" + sentence * "_sentence" + ".csv"
	try:
		df = pd.DataFrame.from_csv(chemin)
	except IOError:
		print("Missing file")
		if sentence:
			df = create_init_df_sentence()
		else:
			df = create_init_df()
	columns = list(df)
	print("Number of missing values...")
	for col in columns:
		print(col,len(df[col]) - df[col].count())

	if not sentence:
		print("Extracting keywords...")
		df["text"] = df["text"].apply(extract_keywords)

	print("Vectorizing content...")
	try:
		model = fst.load_model("model.bin")
	except ValueError:
		print("Model FastText Missing, add path to wikipedia corpus")
		model = fasttext.skipgram('...', 'model') # add path to wikipedia corpus



	dim_vectors = 100
	print("Model loaded")
	df["vector_content"] = df["text"].apply(lambda x:vectorize(x, model))

	

	content_vector = ["content_vector_" + str(i) for i in range(dim_vectors)]

	for i in range(dim_vectors):
		df[content_vector[i]] = df["vector_content"].apply(lambda x:x[i])

	df = df.fillna(0)

	columns_to_drop = ["vector_content"]

	return df.drop(columns_to_drop, axis=1)





def vectorize(x, model):
	""" Loads a FastText model and average the vector representations of 
	words in x"""
	if not x:
		return np.array([0.]*100)

	words = x.split(" ")
	vector = np.array([0.]*100)
	vector = sum([np.array(model[word]) for word in words])

	return vector / len(words)



def extract_keywords(text):
	""" Extracts the keywords from a text"""

	my_tokens = clean_text_simple(text)
	try:
		g = terms_to_graph(my_tokens, w=3)
	except IndexError:
		return None
	# Degrees of the words in the text (dictionnary)
	core_numbers = unweighted_k_core(g)
	# Max degree
	max_c_n = max(core_numbers.values())

	# keywords: words whose degree is max_degree
	keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
	text = ' '.join(keywords)
	return text
