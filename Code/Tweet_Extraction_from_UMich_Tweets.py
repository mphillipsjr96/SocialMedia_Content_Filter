#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[37]:


import pandas as pd
from tqdm import tqdm


# In[38]:


full_data = pd.read_csv('Desktop/full_tweets_umich.csv')


# In[39]:


full_tweets = full_data['text']


# In[41]:


from langdetect import detect


# In[42]:


langdetect_holder = []
for doc in tqdm(full_tweets):
    try:
        if detect(doc) == 'en':
            langdetect_holder.append(doc)
    except:
        continue


# In[199]:


from vaderSentiment import vaderSentiment


# In[200]:


analyzer = vaderSentiment.SentimentIntensityAnalyzer()


# In[201]:


vader_tweets = []
sentiment = []

for tweet in tqdm(langdetect_holder):
    scores = analyzer.polarity_scores(tweet)
    if (scores['pos'] >= .7):
        vader_tweets.append(tweet)
        sentiment.append('positive')
        continue
    if (scores['neg'] >= .7):
        vader_tweets.append(tweet)
        sentiment.append('negative')
        continue
    if (scores['neu'] >= .7):
        vader_tweets.append(tweet)
        sentiment.append('neutral')
        continue


# In[232]:


import numpy as np


# In[233]:


data = np.array([vader_tweets, sentiment]).T
vader_df = pd.DataFrame(data = data, columns = ['tweets', 'sentiment'])


# In[235]:


vader_df.to_csv('Desktop/vader_tweets_.7.csv', index=False)


# In[237]:


vader_df_filtered = vader_df[(~vader_df['tweets'].str.contains('weather')) & (~vader_df['tweets'].str.contains('🔥'))]
vader_df_filtered = vader_df_filtered[~vader_df['tweets'].str.contains(
    'Posted without comment because absolutely no comment is needed')]


# In[238]:


vader_df_filtered['tweets'].apply(len).median()
#median of 101 characters


# In[251]:


vader_df_short = vader_df_filtered[vader_df_filtered['tweets'].apply(lambda x : len(x) < 101)]
vader_df_long = vader_df_filtered[vader_df_filtered['tweets'].apply(lambda x : len(x) > 101)]


# In[252]:


short_neu = vader_df_short[vader_df_short['sentiment']=='neutral']
short_pos = vader_df_short[vader_df_short['sentiment']=='positive']

long_neu = vader_df_long[vader_df_long['sentiment']=='neutral']
long_pos = vader_df_long[vader_df_long['sentiment']=='positive']

short_neu_tweets = short_neu.sample(n=13, random_state=3)
long_neu_tweets = long_neu.sample(n=12, random_state=3)

short_pos_tweets = short_pos.sample(n=13, random_state=3)
long_pos_tweets = long_pos.sample(n=12, random_state=3)


# In[253]:


short_neg = vader_df_short[vader_df_short['sentiment']=='negative']
long_neg = vader_df_long[vader_df_long['sentiment']=='negative']
short_neg_tweets = short_neg.sample(n=37, random_state=3)
long_neg_tweets = long_neg.sample(n=13, random_state=3)


# In[254]:


sampled_tweets = pd.concat(
    [short_neu_tweets, long_neu_tweets, short_pos_tweets,long_pos_tweets,
     short_neg_tweets,long_neg_tweets]).sample(frac=1, random_state=42)


# In[256]:


sampled_tweets.to_csv('Desktop/sampled_tweets.csv', index=False)

