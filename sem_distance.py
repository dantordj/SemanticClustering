from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import time

 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
 
    # Tokenize and tag
    sentence1 = word_tokenize(sentence1)
    sentence2 = word_tokenize(sentence2)

    # Get the synsets for the tagged words
    synsets1 = [wn.synsets(word)[0] for word in sentence1 if wn.synsets(word)]
    synsets2 = [wn.synsets(word)[0] for word in sentence2 if wn.synsets(word)]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    score /= count
 

    return score
 



def sem_distance(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return 1 - (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 

