#importing libraries
import pandas as pd
import re
import preprocessor as p

#removing emojis, URLs, Hashtags, etc.
def preprocess(row):
    text = row['Full_Text']
    text = p.clean(text)
    return text

#additional text cleaning
def TextClean(column):
    column = column.apply(lambda x: re.sub('@[^\s]+', '', str(x)))
    column = column.apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    column = column.apply(lambda x: x.lower())
    return column