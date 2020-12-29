import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize        
import re
import unicodedata
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#text preprocessor with lemmatization
class LemmaTokenizer(object):
    def __init__(self):
        self.lemma = WordNetLemmatizer()
    def __call__(self, tweets):
        return [self.lemma.lemmatize(t) for t in word_tokenize(tweets)]
    
#text preprocessor with stemming
class StemTokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
    def __call__(self, tweets):
        return [self.stemmer.stem(t) for t in word_tokenize(tweets)]
    
#fitting training model
def TrainModel(model, X, Y):
    model.fit(X, Y)
    
    print("Best estimator:\n{}".format(model.best_estimator_))
    print("\nLogistic regression step:\n{}".format(model.best_estimator_.named_steps["clf"]))
    print("\nBest cross-validation score: {:.2f}".format(model.best_score_))
    print("\nBest parameters:\n{}".format(model.best_params_))
    
    results = pd.DataFrame(model.cv_results_)
    return display(results.head())

#model coefficients viz
def CoefViz(model):
    vectors = model.best_estimator_.named_steps['vect']
    feature_names = np.array(vectors.get_feature_names())
    coefficient = model.best_estimator_.named_steps['clf'].coef_ 
    return mglearn.tools.visualize_coefficients(coefficient, feature_names, n_top_features = 40)

#extracting scores from grid_search
def gsHeatmap(model, param_grid, clf_param):
    scores = model.cv_results_['mean_test_score'].reshape(-1, 3).T 
    
    plt.figure(figsize=(15,8))
    model_heatmap = mglearn.tools.heatmap(scores, xlabel = "C", ylabel = "ngram_range",
                                       cmap = "PuBu", fmt = "%.3f", xticklabels = param_grid[clf_param],
                                       yticklabels = param_grid['vect__ngram_range'])
    return plt.colorbar(model_heatmap)

#returning features with lowest and highest tfidf
def MinMaxTfidf(model):
    #seeing which words are listed as "most important" 
    vectorizer = model.best_estimator_.named_steps["vect"] 

    #transform the training dataset
    X_train_model = vectorizer.transform(X_train)

    #find maximum value for each of the features over the dataset 
    max_value = X_train_model.max(axis=0).toarray().ravel() 
    sorted_by_tfidf = max_value.argsort()

    #get feature names
    feature_names = np.array(vectorizer.get_feature_names()) 
    
    print("Features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]])) 
    print("Features with highest tfidf: \n{}".format(feature_names[sorted_by_tfidf[-20:]]))

#testing model
def ModelResults(model, X, Y):
    model_predict = model.predict(X)
    np.mean(model_predict == Y)
    
    print("Test score: {:.2f}".format(model.score(X, Y)))
    print(confusion_matrix(Y, model_predict))
    print(classification_report(Y, model_predict))
