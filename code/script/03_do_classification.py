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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score
import sys 
import os
sys.path.append(os.path.abspath("/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/code/script/functions"))
from funs_do_classification import *

random.seed(101)

#import clean data
clean_df = pd.read_csv('/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/data/processed_data/mi_twitter_data_clean.csv')

#adding custom stop words
stop_words = text.ENGLISH_STOP_WORDS.union('thing', 'la', 'd', 'gon', 'na', 'wa', 'dont' 
                                           'padukone', 'ha', 'le', 'u', 'youre', 'im', 'thats', "ive")

#splitting data into train & test sets
X = clean_df['Clean_Text']
y = clean_df['MI_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

print("Number of tweets in training data: {}".format(len(X_train))) 
print("Samples per class (training): {}".format(np.bincount(y_train)))
print("Number of tweets in test data: {}".format(len(X_test))) 
print("Samples per class (test): {}".format(np.bincount(y_test)))



##############################  
# Naive Bayes Classifier
############################## 

#text normalization, vectorization & classification pipeline
nb_pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = LemmaTokenizer(), 
                             min_df = 5, stop_words = stop_words)),
    ('tfid', TfidfTransformer()),
    ('clf', MultinomialNB(fit_prior = False))
])

#parameter grid with alpha and ngram ranges
nb_param_grid = {
    "clf__alpha": [0.01, 0.1, 0.5, 1.0, 10.0],
    "vect__ngram_range": [(1, 1), (2, 2), (3, 3)]}

nb_grid = GridSearchCV(nb_pipeline, nb_param_grid, cv = 5)

TrainModel(nb_grid, X_train, y_train)
CoefViz(nb_grid)
gsHeatmap(nb_grid, nb_param_grid)

#fitting model to test data
ModelResults(nb_grid, X_test, y_test)




###################################  
# Logistic Regression Classifier
##################################

lr_pipeline = Pipeline([('vect', CountVectorizer(tokenizer = LemmaTokenizer(), 
                             min_df = 5, stop_words = stop_words)), 
                        ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression(max_iter = 1000))
                       ])

lr_param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "vect__ngram_range": [(1, 1), (2, 2), (3, 3)]}

lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv = 5)

TrainModel(lr_grid, X_train, y_train)
CoefViz(lr_grid)
gsHeatmap(lr_grid, lr_param_grid)

#fitting model to test data
ModelResults(lr_grid, X_test, y_test)




###################################  
# Linear SVC Classifier
##################################

# text normalization, vectorization & classification pipeline
svc_pipeline = Pipeline([('vect', CountVectorizer(tokenizer = LemmaTokenizer(), 
                                                 min_df = 5, stop_words = stop_words)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                    ])

# building parameter grid with C and ngram ranges
svc_param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "vect__ngram_range": [(1, 1), (2, 2), (3, 3)]}

svc_grid = GridSearchCV(svc_pipeline, svc_param_grid, cv = 5)

TrainModel(svc_grid, X_train, y_train)
CoefViz(svc_grid)
gsHeatmap(svc_grid, svc_param_grid)

#fitting model to test data
ModelResults(svc_grid, X_test, y_test)




###################################  
# Random Forest Classifier
##################################

rf_pipeline = Pipeline([('vect', CountVectorizer(tokenizer = LemmaTokenizer(), 
                                                 min_df = 5, stop_words = stop_words)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier()),
                    ])

rf_param_grid = {
    "clf__criterion": ['gini', 'entropy'],
    "vect__ngram_range": [(1, 1), (2, 2), (3, 3)]}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv = 5)


TrainModel(rf_grid, X_train, y_train)
gsHeatmap(rf_grid, rf_param_grid)

#fitting model to test data
ModelResults(rf_grid, X_test, y_test)




###################################  
# XGBOOST Classifier
##################################

xgb_pipeline = Pipeline([('vect', CountVectorizer(tokenizer = LemmaTokenizer(), 
                                                 min_df = 5, stop_words = stop_words)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', xgb.XGBClassifier()),
                    ])

xgb_param_grid = {
    "clf__subsample": [0.7, 0.8, 0.9],
    "vect__ngram_range": [(1, 1), (2, 2), (3, 3)]}

xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv = 5, n_jobs=-1)


TrainModel(xgb_grid, X_train, y_train)
gsHeatmap(xgb_grid, rf_param_grid)

#fitting model to test data
ModelResults(xgb_grid, X_test, y_test)


###################################  
# Dataframe Comparing Classifiers
##################################

#naive bayes prediction metrics
nb_prediction = nb_grid.predict(X_test)

nb_accuracy_metric = accuracy_score(y_test, nb_prediction)
nb_precision = precision_score(y_test, nb_prediction, average = "macro")
nb_recall_metric = recall_score(y_test, nb_prediction, average = "macro")
nb_f1_metric = f1_score(y_test, nb_prediction, average = "macro")

#logistic regression prediction metrics
lr_prediction = lr_grid.predict(X_test)

lr_accuracy_metric = accuracy_score(y_test, lr_prediction)
lr_precision = precision_score(y_test, lr_prediction, average = "macro")
lr_recall_metric = recall_score(y_test, lr_prediction, average = "macro")
lr_f1_metric = f1_score(y_test, lr_prediction, average = "macro")

#linear svc prediction metrics
svc_prediction = svc_grid.predict(X_test)

svc_accuracy_metric = accuracy_score(y_test, svc_prediction)
svc_precision = precision_score(y_test, svc_prediction, average = "macro")
svc_recall_metric = recall_score(y_test, svc_prediction, average = "macro")
svc_f1_metric = f1_score(y_test, svc_prediction, average = "macro")

#random forest prediction metrics
rf_prediction = rf_grid.predict(X_test)

rf_accuracy_metric = accuracy_score(y_test, rf_prediction)
rf_precision = precision_score(y_test, rf_prediction, average = "macro")
rf_recall_metric = recall_score(y_test, rf_prediction, average = "macro")
rf_f1_metric = f1_score(y_test, rf_prediction, average = "macro")

#xgboost prediction metrics
xgb_prediction = xgb_grid.predict(X_test)

xgb_accuracy_metric = accuracy_score(y_test, xgb_prediction)
xgb_precision = precision_score(y_test, xgb_prediction, average = "macro")
xgb_recall_metric = recall_score(y_test, xgb_prediction, average = "macro")
xgb_f1_metric = f1_score(y_test, xgb_prediction, average = "macro")

#creating comparison dataframe
models_score_df = pd.DataFrame({'naive_bayes':[nb_accuracy_metric,
                                              nb_precision,
                                              nb_recall_metric,
                                              nb_f1_metric],
                                'logistic_regression':[lr_accuracy_metric,
                                              lr_precision,
                                              lr_recall_metric,
                                              lr_f1_metric],
                                'svc':[svc_accuracy_metric,
                                              svc_precision,
                                              svc_recall_metric,
                                              svc_f1_metric],
                                'random_forest':[rf_accuracy_metric,
                                              rf_precision,
                                              rf_recall_metric,
                                              rf_f1_metric],
                                'xgboost':[xgb_accuracy_metric,
                                              xgb_precision,
                                              xgb_recall_metric,
                                              xgb_f1_metric]},
                               
                               index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

models_score_df['best_score'] = models_score_df.idxmax(axis=1)

models_score_df
