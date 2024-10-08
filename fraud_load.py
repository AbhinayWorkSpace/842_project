import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
# I'm not sure if we want to use these, need to look into it a little more still
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models import Word2Vec
import string
import re

def feature_engineer(df):
    def get_pos_counts(text):
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return pd.Series(dict(pd.DataFrame(pos_tags)[1].value_counts()))

    def flesch_kincaid_grade(text):
        sents = sent_tokenize(text)
        words = word_tokenize(text)
        num_syllables = sum([count_sylls(w) for w in words])

        if len(sents) == 0 or len(words) == 0:
            return 0

        return 0.39 * (len(words) / len(sents)) + 11.8 * (num_syllables / len(words)) - 15.59

    # This definitely needs to be checked
    def count_sylls(word):
        return len(
            re.findall('(?!e$)[aeiouy]+', word.lower(), re.I) +
            re.findall('^[^aeiouy]*e$', word.lower(), re.I)
        )

    dl_path = './nltk_data'
    nltk.download('punkt', download_dir=dl_path)
    nltk.download('punkt_tab', download_dir=dl_path)
    nltk.download('wordnet', download_dir=dl_path)
    nltk.download('stopwords', download_dir=dl_path)
    nltk.download('averaged_perceptron_tagger', download_dir=dl_path)
    nltk.download('averaged_perceptron_tagger_eng', download_dir=dl_path)
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
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
    df['avg_sentence_length'] = df['text'].apply(lambda x: np.mean([len(sent.split()) for sent in sent_tokenize(x)]))
    df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in x if c in string.punctuation]))
    # df['stopword_count'] = df['text'].apply(lambda x: len([w for w in x.split() if w in stopwords.words('english')]))
    df['punctuation_ratio'] = df['punctuation_count'] / df['text_length']
    df['uppercase_ratio'] = df['text'].apply(lambda x: len([c for c in x if c.isupper()])) / df['text_length']

    pos_counts = df['text'].apply(get_pos_counts)
    pos_feats = pos_counts.add_prefix('pos_')
    df = pd.concat([df, pos_feats], axis=1)

    df['flesch_kincaid_grade'] = df['text'].apply(flesch_kincaid_grade)

    # sentiment analysis
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)


    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    tfidf_feature_names = tfidf.get_feature_names_out()
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
    df = pd.concat([df, tfidf_features], axis=1)

    return df

def modelWord2Vec(text):
    # Gensim documentation: https://radimrehurek.com/gensim/models/word2vec.html
    # Create a Word2Vec model based on our pre-processed data
    w2v_model = Word2Vec(sentences=text, vector_size=100, window=5, min_count=1, workers=4)
    # Save a Word2Vec model
    w2v_model.save("word2vec.model")
    # Load a Word2Vec model
    w2v_model = Word2Vec.load("word2vec.model")
    # When we want to train the model, we'll use model.train()
    # model.train()

def load_df():
    df = pd.read_csv('fraud.csv')
    df = feature_engineer(df)
    return df

def get_engineered_data():
    engineered = load_df()
    modelWord2Vec(engineered)
    engineered.to_csv('fraud_engineered.csv', index=False)
    return engineered

get_engineered_data()