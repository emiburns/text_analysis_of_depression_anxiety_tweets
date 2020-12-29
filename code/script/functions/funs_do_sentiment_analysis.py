#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import mglearn
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import gensim
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim import corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

stopwords = set(stopwords.words('english'))
newStopWords = ["youre", "im", "thats", "bc", "ive"]
stopwords = stopwords.union(newStopWords)

#wordcloud visualization
def WordCloudViz(corpora):
    wordcloud = WordCloud(stopwords = stopwords, background_color = "black", 
                      max_words = 150).generate(corpora)
    fig = plt.figure(1, figsize=(12, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt.show()

#defining tweet preprocessing function w/ nltk and gensim
def preprocess_tweets(df):
    corpus=[]
    stem = SnowballStemmer("english")
    lem = WordNetLemmatizer()
    for tweet in df['Clean_Text']:
        words = [w for w in word_tokenize(tweet) if (w not in stopwords)]
        words = [lem.lemmatize(w) for w in words if len(w)>2]
        corpus.append(words)
        
    return corpus

#topic modeling
def TopicModeling(corpus):
    dep_dict = corpora.Dictionary(corpus)
    dep_bow = [dep_dict.doc2bow(doc) for doc in corpus]
    
    dep_lda_model = gensim.models.LdaMulticore(dep_bow, 
                                   num_topics = 10, 
                                   id2word = dep_dict,                                    
                                   passes = 10,
                                   workers = 2)
    return dep_lda_model.show_topics()