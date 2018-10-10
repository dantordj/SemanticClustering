import pandas as pd
import os
import string
from nltk.corpus import stopwords
import re

punct = string.punctuation.replace('-', '')

def clean(text):
    clean_text = text.replace('"', '\"')
    clean_text = clean_text.replace("``", "")
    clean_text = clean_text.replace("/.", "")
    clean_text = clean_text.replace("''", "")
    clean_r = re.compile('<.*?>')
    clean_re = re.compile('/.*? ')
    clean_tab = re.compile(r'[\n\r\t]')
    printable = set(string.printable)
    clean_text = re.sub(clean_r, '', clean_text)
    clean_text = re.sub(clean_re, ' ', clean_text)
    clean_text = re.sub(clean_tab, ' ', clean_text)
    clean_space = re.compile('[ ][ ]*')
    clean_text = re.sub(clean_space, ' ', clean_text)
    for sign in punct:
        clean_text = clean_text.replace(" " + sign, sign)

    return filter(lambda x: x in printable, clean_text).strip()

def create_init_df():

    nct = string.punctuation.replace('-', '')
    values = {"text": [], "real_cluster":[]}
    i = 0
    cluster_to_int = {}
    count_clusters = 0
    for element in os.listdir('brown'):
        if element == '.DS_Store':
            continue
        cluster_str = element[0:2]
        try:
            cluster = cluster_to_int[cluster_str]
        except KeyError:
            cluster_to_int[cluster_str] = count_clusters
            count_clusters += 1
            cluster = cluster_to_int[cluster_str]
        with open('brown/' + element) as fil:
            text = fil.read()
            # clean text
            text = clean(text)
            i += 1
            values["text"] += [text]
            values["real_cluster"] += [cluster]
    df = pd.DataFrame(values)

    print("Num clusters", count_clusters)

    print(cluster_to_int)
    df.to_csv("df_brown.csv")
    return df


def create_init_df_sentence():
    values = {"text": [], "real_cluster":[]}
    i = 0
    cluster_to_int = {}
    count_clusters = 0
    for element in os.listdir('brown'):
        if element == '.DS_Store':
            continue
        cluster_str = element[0:2]
        try:
            cluster = cluster_to_int[cluster_str]
        except KeyError:
            cluster_to_int[cluster_str] = count_clusters
            count_clusters += 1
            cluster = cluster_to_int[cluster_str]
        with open('brown/' + element) as fil:
            text = fil.read()
            text = clean(text)
            text = text.split(". ")
            # clean text
            for sentence in text:
                i += 1
                values["text"] += [sentence]
                values["real_cluster"] += [cluster]
    df = pd.DataFrame(values)
    df.to_csv("df_brown_sentence.csv")
    return df
    