import pandas as pd
from gensim.utils import simple_preprocess
from gensim import corpora
import nltk
import string
from nltk.corpus import stopwords, words
import os
import hashlib
import json


lemmatizer = nltk.stem.WordNetLemmatizer()
dictionary = corpora.Dictionary()

english_dictionary = set(words.words())

if os.path.exists('stop_words.txt'):
    with open('stop_words.txt') as f:
        stopwords = []
        for line in f:
            stopwords.append(line)

else:
    custom_words = [
                       'none', '. ', 'ie', 'etc', 'eg', 'also',
                   ] + list(string.ascii_lowercase) + list(string.digits)

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('words')
    nltk.download('omw-1.4')
    punctuation = string.punctuation.replace('.', '').replace('-', '')
    stop_words = set(stopwords.words("english") + list(punctuation) + custom_words)
    file = open('stop_words.txt', 'w+')
    file.write('\n'.join(stop_words))
    file.close()

with open('categories.json') as f:
    categories = json.load(f)


def check_arXiv_metdata(metadata_file):

    # Check arXiv metadata directory exists
    if not os.path.exists('../arxiv_metadata'):
        os.mkdir('../arxiv_metadata')
        return False

    if not os.path.exists('//arxiv_metadata/arxiv-metadata-oai-snapshot.json'):
        return False

    # Compare the newly downloaded arxiv file to the existing file to see if updates need to be made
    new_file_hash = hashlib.sha256(open(metadata_file, 'rb').read()).hexdigest()
    old_file_hash = hashlib.sha256(open('//arxiv_metadata/arxiv-metadata-oai-snapshot.json', 'rb').read()).hexdigest()

    if new_file_hash == old_file_hash:
        return True

    return False


def get_category_keywords(text):

    category_keywords = []
    for category in text.split(' '):
        if category.__contains__('.'):
            main, sub = category.split('.')
            if main in categories:
                if sub in categories[main]:
                    category_keywords += categories[main]['desc']
                    category_keywords += categories[main][sub]
        else:
            if category in categories:
                category_keywords += categories[category]['desc']

    return category_keywords


def extract_keywords(row):

    #authors = row["authors"]
    title = row["title"]
    comments = row["comments"]
    row_categories = row["categories"]
    abstract = row["abstract"]

    category_keywords = get_category_keywords(row_categories)

    row["keywords"] = simple_preprocess(title + comments + abstract) + category_keywords
    row["keywords"] = [lemmatizer.lemmatize(word) for word in row["keywords"] if (word in english_dictionary and word not in stop_words)]

    return row