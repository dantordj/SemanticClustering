# Semantic clustering

In this repos we implement a recurrent approach to define embeddings for words, sentences and documents. 

For words, sentences and documents, we use the same approach : 
- initilaizion of embeddings with the FastText library (for sentences we take the average vector after stemming, for documents the average vector after keywords extraction)
- k-mean algorithm to cluster the vectors 
- building of a training set with the k% best clustered vectors, testing set with the others
- training of a FFNN 
- definition of new embeddings for words/sentences/documents from the testing set (vectors extracted from the second layer of the FFNN)
- iteration of the process

We evaluate the embeddings we get for documents with a topic visualization (pylda).



Add wikipedia corpus (for fasttext training) to the root Folder and Brown cropus. 

Run main.py for clustering Brown documents and visualizing the results.

Open test_visualization.html in a browser for topic visualization.

## Depencies
- Brown Library (http://www.nltk.org/nltk_data/, Brown Corpus)
- Python 2.7
- Pandas
- Nltk
- Igraph
- FastText 
- Scikit-learn
- Numpy
- Scipy
