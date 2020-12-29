#importing libraries
import pandas as pd
import re
import preprocessor as p
import sys 
import os
sys.path.append(os.path.abspath("/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/code/script/functions"))
from funs_do_data_cleaning import *

#load raw data 
raw_df = pd.read_csv("/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/data/raw_data/mi_twitter_data_raw.csv", index_col = 0)

raw_df.info()
raw_df.head()
raw_df.tail()

#dummy coding classification variable 
raw_df['MI_Type'] = raw_df['MI_Type'].replace('depression', 0)
raw_df['MI_Type'] = raw_df['MI_Type'].replace('anxiety', 1)

#cleaning text
raw_df['Clean_Text'] = raw_df.apply(preprocess, axis=1) 
raw_df['Clean_Text'] = TextClean(raw_df['Clean_Text'])

#finding and removing instances of "great depression" (confounding info)
raw_df['Clean_Text'].str.contains('great depression').sum() #1334 instances
raw_df = raw_df[~raw_df['Clean_Text'].str.contains('great depression')]
raw_df.info()

#checking for NA values
raw_df.isna().sum()
raw_df[raw_df['Clean_Text'].isnull()].index
raw_df = raw_df.dropna()

#creating new column for number of words in a tweet
raw_df['Text_Length'] = raw_df['Clean_Text'].apply(len)

#writing raw data to csv file
tweets_clean = raw_df
path = '/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/data/processed_data'
output_file = os.path.join(path,'mi_twitter_data_clean.csv')
tweets_clean.to_csv(output_file, index = False)