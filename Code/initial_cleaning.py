# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:40:19 2021

@author: micha
"""
#%% Imports
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from gensim.models.phrases import Phrases, Phraser

#%% Global Variables
CUSTOM_STOP_WORDS = ['www','twitpic','tinyurl','com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
num_topics = 10

#%% Initial Load
tweet_df = pd.read_csv("/Data/training.1600000.processed.noemoticon.csv",encoding='latin-1',names=['target','ids','date','flag','user','text'])

#%% Tokenization
tweet_df['text'] = tweet_df['text'].astype(str)
tweet_df['tokens'] = tweet_df['text'].apply(lambda x: preprocess_string(x))
FULL_STOP = STOPWORDS.union(set(CUSTOM_STOP_WORDS))
tweet_df['tokens'] = tweet_df['tokens'].apply(lambda x: [word for word in x if word not in (FULL_STOP)])

#%% Bigrams
phrase_model = Phrases(tweet_df['tokens'],min_count=20,threshold=2).freeze()
tweet_df['tokens']=tweet_df['tokens'].apply(lambda x: phrase_model[x])

tweet_df.to_csv("/Data/bigram_tweet_df.csv",index=False)

#%% Function



def preprocessing(phrase_model,tweet,stopwords):

    tweet_tokens = preprocess_string(tweet['body'])
    tweet_tokens = [word for word in tweet_tokens if word not in (stopwords)]

    return phrase_model[tweet_tokens]


preprocessing(phrase_model,
              {"body":"I love visiting New York and the Statue of Liberty"},
              FULL_STOP)


