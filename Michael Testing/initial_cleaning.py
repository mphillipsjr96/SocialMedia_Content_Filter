# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:40:19 2021

@author: micha
"""
#%% Imports
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel

#%% Global Variables
CUSTOM_STOP_WORDS = ['www','twitpic','tinyurl','com', 'https', 'http', '&amp', 'rt', 'bit', 'ly', 'bitly']
num_topics = 10

#%% Initial Load
tweet_df = pd.read_csv("D:/Documents/Python/SocialMedia_Content_Filter/Data/training.1600000.processed.noemoticon.csv",encoding='latin-1',names=['target','ids','date','flag','user','text'])

#%% Tokenization
tweet_df['text'] = tweet_df['text'].astype(str)
tweet_df['tokens'] = tweet_df['text'].apply(lambda x: preprocess_string(x))
FULL_STOP = STOPWORDS.union(set(CUSTOM_STOP_WORDS))
tweet_df['tokens'] = tweet_df['tokens'].apply(lambda x: [word for word in x if word not in (FULL_STOP)])

#%% Bigrams
phrase_model = Phrases(tweet_df['tokens'],min_count=20,threshold=2).freeze()
tweet_df['tokens']=tweet_df['tokens'].apply(lambda x: phrase_model[x])
tweet_df_sample = tweet_df.head(5)

#%% Topics from 682 HW 4
def find_topics(tokens, num_topics):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=10,no_above=.6)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    lda_model = LdaModel(corpus=corpus,id2word = dictionary,num_topics=num_topics,eval_every=None,random_state=42,alpha='auto',eta='auto')
    return lda_model.top_topics(corpus) 

def calculate_avg_coherence(topics):
    avg_topic_coherence = None
    avg_topic_coherence = sum([t[1] for t in topics]) / len(topics)
    return avg_topic_coherence

def plot_coherences_topics(tokens):
    topics_range = range(2, 11, 1)
    model_results = {'Topics': [],'Coherence': []}
    from tqdm import tqdm
    for topic in tqdm(topics_range):
        tops = find_topics(tokens, topic)
        model_results['Topics'].append(topic)
        model_results['Coherence'].append(calculate_avg_coherence(tops))
    plt = pd.DataFrame(model_results).set_index('Topics').plot()

plot_coherences_topics(tweet_df['tokens'])