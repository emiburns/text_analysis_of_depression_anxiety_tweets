Semantic Differences Between Twitter-Based Expressions of Anxiety and
Depression
================

### Project Overview

-----

Anxiety and depression are common and debilitating mental disorders that
can cause significant impairment in daily functioning, physical health,
and cognition. Understanding how symptoms of these clinical disorders
are expressed can aid in early detection, progress of AI-aided virtual
therapy platforms, and general clarification of differential linguistic
properties between anxiety and depression.

This project integrates personal interests of mental health, natural
language processing (NLP), and statistical modeling by utilizing NLP and
supervised learning classification algorithms to analyze and classify
depression and anxiety-oriented tweets pulled by use of the Twitter API.

### Data Source

-----

**Twitter**- 10,000 tweets meeting keyword “depression” and “anxiety”
criteria were pulled utilizing the [Twitter](https://twitter.com/home)
API on September 26, 2020. Aquired text corpus and related tweet
identifiers and characteristics were saved to csv file
‘mi\_twitter\_data\_raw.csv’.

### Project Files

-----

#### General files

  - *README.ipynb*- Current file. Project overview.

  - *codebook.ipynb*- Description of data variable names

#### data/ folder

  - *raw\_data/* - Includes raw csv file as identified in project
    sources section

  - *processed\_data/* - Includes csv file with cleaned tweets used in
    modeling analyses, as well as csv with calculated sentiment analysis
    variables.

#### code/ folder

**exploration/ subfolder** \* Files are numbered in chronological order

  - *01\_data\_exploration.ipynb*: Markdown file of exploratory data
    analysis (EDA) results

  - *02\_sentiment\_analysis.ipynb*: Markdown file of unsupervised
    learning, latent dirichlet allocation, and sentiment analysis
    results

  - *03\_classification\_results.ipynb*: Markdown file of model results
    and their interpretation

**script/ subfolder**

  - *01\_do\_data\_collection.py*- Code used to pull text data via the
    Twitter API and save into raw csv

  - *02\_do\_data\_cleaning.py*- Code used to process text data prior to
    modeling

  - *03\_do\_data\_classification.py*- Code with supervised learning
    algorithms used to predict whether a tweet was depression or
    anxiety-oriented. Models used in analyses include naive bayes,
    logistic regression, linear SVC, random forest and xgboost. See
    section below for results overview

  - *functions/ subfolder*- Includes functions utilized for data
    cleaning (funs\_do\_data\_cleaning.py), EDA
    (funs\_do\_exploratory\_data\_analysis.py), sentiment analysis
    (funs\_do\_sentiment\_analysis.py) and classification
    (funs\_do\_classification.py)

### Project Results

-----

Tweets were identified as either ‘positive’ or ‘negative’ sentiment,
with more depression oriented tweets labeled as negative more often than
anxiety-oriented. Tweets were tokenized with lemmatization and removal
of stopwords and then transformed with tfidf weighting scheme prior to
fitting to supervised learning algorithms. Random forest ensemble
classifier with gini parameter specification and single ngrams produced
the highest model accuracy, predicting tweets as either depression or
anxiety oriented at 97% accuracy.

Findings support the ability of NLP to accurately identify anxiety vs
depression related posts on Twitter, despite the high clinical
comorbidity of both symptoms. This, coupled with the differences in
topic sentiment analysis, suggest that there are distinct linguistic
differences in how depression and anxiety are verbalized on social
media.
