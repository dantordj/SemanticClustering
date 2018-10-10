import itertools
import re
import copy
import igraph
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from nltk.corpus import stopwords


def clean_text_simple(text):
    """ return array of words depending on parameters. 
    pos_filtering: select part of speech tags to keep
    remove_stopwords: remmove non-meaningful stopwords such as "can", "will"
    stemming: transform words into their radical form ("winning" --> "win")
    """
    # Words to remove from the text because they are meaningless of punctuation
    my_stopwords = stopwords.words('english')
    punct = string.punctuation.replace('-', '')

    text = text.lower()
    text = ''.join(l for l in text if l not in punct)  # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +', ' ', text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space
    # tokenize (split based on whitespace)
    tokens = text.split(' ')

    # POS tag and retain only nouns and adjectives
    tagged_tokens = pos_tag(tokens)
    tokens_keep = []
    for item in tagged_tokens:
        if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
        ):
            tokens_keep.append(item[0])
    tokens = tokens_keep
   
    # remove stopwords
    tokens = [token for token in tokens if token not in my_stopwords]

    # apply stemmer
    stemmer = nltk.stem.PorterStemmer()
    tokens_stemmed = list()
    for token in tokens:
        tokens_stemmed.append(stemmer.stem(token))
    tokens = tokens_stemmed

    return (tokens)


def terms_to_graph(terms, w):
    """Returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'.
    """
    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i]  # term to consider
        terms_temp = terms[(i - w + 1):(i + 1)]  # all terms within sliding window

        # edges to try
        candidate_edges = []
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            if try_edge[1] != try_edge[0]:
                # if not self-edge

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertex weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return (g)


def unweighted_k_core(g):
    """ return the degree of each vertices as a dictionnary"""
    # work on clone of g to preserve g
    gg = copy.deepcopy(g)

    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'], [0] * len(gg.vs)))

    i = 0

    # while there are vertices remaining in the graph
    while len(gg.vs) > 0:
        # while there is a vertex with degree less than i
        while [deg for deg in gg.strength() if deg <= i]:
            index = [ind for ind, deg in enumerate(gg.strength()) if deg <= i][0]
            # assign i as the matching vertices' core numbers
            cores_g[gg.vs[index]['name']] = i
            gg.delete_vertices(index)  # incident edges on the deleted vertex are automatically removed

        i += 1

    # ou alors cores_g = dict(zip(gg.vs['name'], g.coreness))

    return cores_g
