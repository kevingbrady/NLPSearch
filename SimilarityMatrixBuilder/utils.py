import tensorflow as tf
import numpy as np
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import string

custom_words = [
    'none', '. ', 'ie', 'etc', 'eg'
] + list(string.ascii_lowercase) + list(string.digits)

nltk.download('stopwords')
punctuation = string.punctuation.replace('.', '').replace('-', '')
stop_words = set(stopwords.words("english") + list(punctuation) + custom_words)
lemmatizer = nltk.stem.WordNetLemmatizer()


def generate_keywords(string):
    words = simple_preprocess(string)
    keywords = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    print(keywords)
    return keywords