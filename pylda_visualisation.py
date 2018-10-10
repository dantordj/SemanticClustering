from glob import glob
import re
import string
import funcy as fp
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import TfidfModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import matplotlib
import IPython.display
import keyword_extraction

#works in python 2.7
def load_doc((idx, tokens), kmean_result_path="clusters.csv"):
    clustering_result_df = pd.read_csv(kmean_result_path)
    return {'real_cluster': clustering_result_df['real_cluster'][idx],
            'tokens': tokens.split(' '),
            'id': idx}




def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

def prep_corpus(docs, additional_stopwords=set(), no_below=0, no_above=1):
  print('Building dictionary...')
  dictionary = Dictionary(docs)
  # Filtering the dictionary
  # stopwords = nltk_stopwords().union(additional_stopwords)
  # stopword_ids = map(dictionary.token2id.get, stopwords)
  # dictionary.filter_tokens(stopword_ids)
  # dictionary.compactify()
  # dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
  # dictionary.compactify()

  print('Building corpus...')
  corpus = [dictionary.doc2bow(doc) for doc in docs]

  return dictionary, corpus

def clean(text):
    tokens = keyword_extraction.clean_text_simple(text)
    return ' '.join([token for token in tokens])

def pylda_visualize(csv_chemin, ecriture_chemin, tfidf_visualization = False, num_topic=3, filter_by_cluster=None):
    ''' gets the clustering result from csv_chemin and then writes the LDA visualisation as an html file into ecriture_chemin
        csv_chemin points to a dataframe with two columns: one corresponding to the cluster, the other containing the text
         num_topic is the number of topics we want to extract from the texts
         filter_by_cluster is the cluster index, if we want to extract topics from one cluster only
    '''
    #df = pd.read_csv('df_brown.csv')
    clustering_result_df = pd.read_csv(csv_chemin)
    if filter_by_cluster:
        clustering_result_df[clustering_result_df['pred_cluster'] == filter_by_cluster]
    text = clustering_result_df['text'].values
    #text = ' '.join(text)

    docs = pd.DataFrame(list(map(load_doc, enumerate(list(clustering_result_df['text'].apply(clean))))))
    docs.head()

    dictionary, corpus = prep_corpus(docs['tokens'])
    #dictionary : keys = word_id ; value = word
    #corpus[i] = list of tuples (word_id, count) where count is the number of occurence of the word in the text corpus[i]

    if tfidf_visualization:
        # Instead of representing each text as tuples (word_idx, term_frequency), we represent them as (word_idx, word_tfidf_weight)
        model = TfidfModel(corpus)
        new_corpus = []
        for i in range(len(corpus)):
            element = corpus[i]
            new_element = []
            for j in range(len(element)):
                #word = dictionary[pair[0]]
                pair = element[j]
                #dict_idx = pair[0]
                tfidf_vector = model[element]
                word_tfidf_weight = tfidf_vector[j]
                new_element += (pair[0], word_tfidf_weight)
            new_corpus.append(new_element)

        MmCorpus.serialize(ecriture_chemin + '.mm', corpus)
        dictionary.save(ecriture_chemin + '.dict')

        lda = models.ldamodel.LdaModel(corpus=new_corpus, id2word=dictionary, num_topics=15, passes=10)

        lda.save(ecriture_chemin + '.model')


        vis_data = gensimvis.prepare(lda, new_corpus, dictionary)
        pyLDAvis.display(vis_data)
        pyLDAvis.save_html(vis_data, ecriture_chemin + '.html')

    else:
        MmCorpus.serialize(ecriture_chemin + '.mm', corpus)
        dictionary.save(ecriture_chemin + '.dict')

        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic, passes=10)

        lda.save(ecriture_chemin + '.model')

        vis_data = gensimvis.prepare(lda, corpus, dictionary)
        pyLDAvis.display(vis_data)
        pyLDAvis.save_html(vis_data, ecriture_chemin + '.html')



#brown categories
#['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned',
# 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

