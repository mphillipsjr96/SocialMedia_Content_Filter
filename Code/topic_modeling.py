# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:44:39 2021

@author: micha
"""

import pandas as pd
from tqdm import tqdm
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS 
from gensim.models.phrases import Phrases, Phraser

#%% Data Loading
tweets_df = pd.read_csv('/Data/full_tweets_umich.csv')

#%% Language Detection
langs = []
for row in tqdm(script_tweets['text']):
    try:
        lang = detect(row)
    except:
        lang = 'error'
    langs.append(lang)
    
script_tweets['lang'] = langs

script_tweets.to_csv('D:/Documents/Python/Data/full_tweets_umich_'+str(script)+'.csv')

#%%

tweets_df_lang = pd.read_csv('D:/Documents/Python/Data/full_tweets_umich_1.csv')
tweets_df_lang = tweets_df_lang.append(pd.read_csv('D:/Documents/Python/Data/full_tweets_umich_9.csv'))
tweets_df_lang_en = tweets_df_lang[tweets_df_lang['lang']=='en']
tweets_df_lang_en.drop('Unnamed: 0',inplace=True,axis=1)
tweets_df_lang_en.to_csv('D:/Documents/Python/Data/en_tweets_umich.csv')

#%% Sentiment
sid_obj = SentimentIntensityAnalyzer()

tweets_df_lang_en['sentiment'] = tweets_df_lang_en['text'].apply(lambda x: sid_obj.polarity_scores(x))

#%% Tokenizing
tweet_df = tweets_df_lang_en
CUSTOM_STOP_WORDS = ['www','tinyurl','com', 'https', 'http','&amp', 'rt', 'bit', 'ly', 'bitly']
tweet_df['text'] = tweet_df['text'].astype(str)
tweet_df['tokens'] = tweet_df['text'].apply(lambda x: preprocess_string(x))
FULL_STOP = STOPWORDS.union(set(CUSTOM_STOP_WORDS))
tweet_df['tokens'] = tweet_df['tokens'].apply(lambda x: [word for word in x if word not in (FULL_STOP)])

#%% Phrasing
phrase_model = Phrases(tweet_df['tokens'],min_count=20,threshold=2).freeze()
tweet_df['tokens']=tweet_df['tokens'].apply(lambda x: phrase_model[x])

#%% topics
from gensim.corpora import Dictionary
from gensim.models import LdaModel

def find_topics(tokens, num_topics):
    dictionary = Dictionary(tokens) # Your code here 
    corpus = [dictionary.doc2bow(token) for token in tokens]
    lda_model = LdaModel(corpus=corpus,id2word = dictionary,num_topics=num_topics,chunksize=2000,passes=20,iterations=400,eval_every=None,random_state=42,alpha='auto',eta='auto') # Your code here
    return lda_model.top_topics(corpus) 

def calculate_avg_coherence(topics):
    avg_topic_coherence = sum([t[1] for t in topics]) / len(topics)
    return avg_topic_coherence


model_results = {'Topics': [],'Coherence': []}
tops = find_topics(tweet_df['tokens'], 10)
model_results['Topics'].append(tops)
model_results['Coherence'].append(calculate_avg_coherence(tops))
