# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:50:29 2021

@author: micha
"""

import pickle
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
import json
import numpy as np 

phrase_model = pickle.load(open('SocialMedia_Content_Filter\Michael Testing\phrasemodel.sav','rb'))

CUSTOM_STOP_WORDS = ['www','twitpic','tinyurl','com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
FULL_STOP = STOPWORDS.union(set(CUSTOM_STOP_WORDS))

def preprocessing(phrase_model,tweet,stopwords):
    tweet_tokens = preprocess_string(tweet['body'])
    tweet_tokens = [word for word in tweet_tokens if word not in (stopwords)]
    return phrase_model[tweet_tokens]

f = open('/SocialMedia_Content_Filter/Michael Testing/thesaurus_lem.json','rb')
thesaurus = json.load(f)

def tweet_contains_filter_words(tweet,thesaurus):
    tweet_tokens = preprocess_string(tweet['body'])
    filter_tokens = preprocess_string(" ".join(tweet['filter_words']))
    new_filter_tokens = []
    #check for thesaurus 
    for token in filter_tokens:
        new_filter_tokens.append(token)
        try:
            new_filter_tokens.extend(thesaurus[token])
        except:
            continue

    return any([word in tweet_tokens for word in new_filter_tokens])
    

#%% Word filtering examples
tweet = {'body':"I heard that part of the city is full of goons", "filter_words":["hood"]}
tweet_contains_filter_words(tweet,thesaurus)
thesaurus['hood']

