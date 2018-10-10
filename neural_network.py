import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from process_data import clean_text_simple


def predict_neural_network(df_train, labels, df_test, target_dimensions, num_clusters, model_ffnn = None):
	""" Neural netword to find new vector representations ind dimension 'target_dimensions'. The neural 
	network is trained to predict the label of the cluster"""
	
	print("Training TFIDF...")
	corpus = pd.concat([df_train["text"], df_test["text"]])
	corpus = corpus.reset_index().values
	corpus = [c[1] for c in corpus]
	
	size_train = df_train.shape[0]
	size_corpus = len(corpus)
	vectorizer = TfidfVectorizer(stop_words="english", min_df=0.1)
	vectorizer.fit(corpus)
	features_TFIDF_train = vectorizer.transform(df_train["text"])
	features_TFIDF_test = vectorizer.transform(df_test["text"])

	
	features_name = vectorizer.get_feature_names()
	dict_word_idx = {features_name[i]: i for i in range(len(features_name))}


	input_dim = features_TFIDF_train.shape[1]
	size_layer = target_dimensions

	# If first iteraiton in the algorithm, needs to set the model
	if not model_ffnn:
		model_ffnn = Sequential()

		## TODO : add a line here that adds the extra layer
		
		model_ffnn.add(Dense(size_layer, activation='relu', input_dim=input_dim))
		model_ffnn.add(Dense(num_clusters, activation='softmax'))
		model_ffnn.compile(
		    loss=keras.losses.categorical_crossentropy,
		    optimizer=keras.optimizers.Adagrad(),
		    metrics=['accuracy']
		)
		model_ffnn.summary()


	model_ffnn.fit(features_TFIDF_train, labels, epochs=20, batch_size=10)

	predictions = model_ffnn.predict(features_TFIDF_test)

	predictions = [np.argmax(p) for p in predictions]

	
	# we build a new model with the activations of the old model
	# this model is truncated after the first layer
	model2 = Sequential()
	model2.add(Dense(size_layer, activation='relu', input_dim=input_dim, weights=model_ffnn.layers[0].get_weights()))
	

	activations = model2.predict(features_TFIDF_test)

	return predictions, activations, model_ffnn

