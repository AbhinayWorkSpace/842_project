import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaMulticore
import string
import re

def feature_engineer(df):
    dl_path = './nltk_data'
    nltk.download('punkt', download_dir=dl_path)
    nltk.download('wordnet', download_dir=dl_path)
    nltk.download('stopwords', download_dir=dl_path)
    nltk.download('averaged_perceptron_tagger', download_dir=dl_path)
    nltk.data.path.append(dl_path)

    # load data and format some columns
    df['label'] = df['label'].map({'OR': 0, 'CG': 1})
    # change label header to 'fraud'
    df = df.rename(columns={'label': 'fraud', 'text_': 'text'})


    # feature engineering

    # XLNet embeddings will be valuable, but that is done in training and not here


    # one hot encoding of category
    # I'm not totally certain if we'll want this but i thought it can't hurt for now
    # if we dont want it we can just drop the column
    df = pd.get_dummies(df, columns=['category'], prefix='category')
    # df.drop(columns=['category'], inplace=True)

    df['normalized_rating'] = df['rating'] / 5.0

    # text features
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)))
    return df

def load_df():
    df = pd.read_csv('fraud.csv')
    df = feature_engineer(df)
    return df
