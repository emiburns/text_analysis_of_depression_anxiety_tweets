#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def TweetLength(column):
    avg = column.mean()
    maximum = column.max()
    minimum = column.min()
    
    return "Longest text length: {0}, Shortest text length: {1}, Average text length: {2}".format(maximum, minimum, round(avg,2))

#various plots for tweet length by MI type
def LengthHist(data, column, hue):
    g = sns.FacetGrid(data, col = column, 
                  hue = hue, palette = 'pastel')
    return g.map(plt.hist, 'Text_Length')

def LengthBoxPlot(data, X, Y):
    sns.set_style("whitegrid")
    return sns.boxplot(x = X, y = Y,
            data = data, palette = 'pastel')

def LengthScatterPlot(data, X, Y1, Y2, hue):
    fig, ax = plt.subplots(1,2)
    sns.scatterplot(data = data, x = X, y = Y1, 
                hue = hue, palette = 'pastel', s = 40, ax = ax[0])
    sns.scatterplot(data = data, x = X, y = Y2, 
                hue = hue, palette = 'pastel', s = 40, ax = ax[1])
    return plt.figure(figsize=(15,8))
    
#heatmap of data
def CorHeatmap(data, group):
    plt.figure(figsize=(10,6))
    correlations = data.iloc[:, 2:].groupby(group).mean()
    return sns.heatmap(correlations.corr(), cmap='PuBu', annot = True)

#vectorizing and ordering the most common words used in tweets
def top_ngrams(corpus, n = None):
    vec = CountVectorizer(ngram_range = (n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

#plotting top 25 trigrams
def PlotTrigrams(trigrams):
    x,y = map(list,zip(*trigrams))
    sns.set_context("paper")
    plt.figure(figsize=(15,8))
    return sns.barplot(x = y, y = x)
