from nytimesarticle import articleAPI
import requests
from bs4 import BeautifulSoup
import string
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd

api = articleAPI('b499930464a9444792114d5857b50f1b')



def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    urls = []
    for i in articles['response']['docs']:
        urls.append(i['web_url'])
    return(urls)

def get_articles(date,query):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    all_urls = []
    for i in range(0,50): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        try:
            articles = api.search(q = query,
               fq = {'source':['Reuters','AP', 'The New York Times']},
               begin_date = date + '0101',
               end_date = date + '1231',
               sort='oldest',
               page = str(i))
            urls = parse_articles(articles)
            all_urls = all_urls + urls
        except KeyError, e:
            print(str(e))
            continue
        except ValueError, e:
            print(str(e))
            continue
    return(all_urls)

def scrape_url(url):
    '''
    :param url:  url to an article web page
    :return:    the article text taken from the page's source code
    '''
    session = requests.Session()
    req = session.get(url)
    soup = BeautifulSoup(req.text)
    paragraphs = soup.find_all('p', class_='story-body-text story-content')
    article = ''
    for p in paragraphs:
        article = article + p.get_text().encode('utf-8')

    return  ' '.join(clean_text_simple(article))

def create_theme_csv(query):
    '''

    :param query: theme of the corpus to be created (eg: sport, art, politics...)
    This method creates a csv file with one column containing each article's text
    '''
    urls = get_articles('2017', query)
    articles = []
    for url in urls:
        articles.append(scrape_url(url))
    df = pd.DataFrame(articles, columns=["article"])
    df.to_csv('nytimes/' + query + '.csv')


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

    # remove stopwords
    tokens = [token for token in tokens if token not in my_stopwords]
    # apply stemmer
    stemmer = nltk.stem.PorterStemmer()
    tokens_stemmed = list()
    for token in tokens:
        tokens_stemmed.append(stemmer.stem(token))
    tokens = tokens_stemmed
    return (tokens)

