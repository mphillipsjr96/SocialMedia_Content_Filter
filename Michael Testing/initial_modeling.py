# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:49:52 2021

@author: micha
"""
#%% Imports
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
 
#%% Global Variables
RANDOM_SEED = 42
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }  
#%% Load Data
tweet_df = pd.read_csv('D:/Documents/Python/Data/bigram_tweet_df.csv')
y = tweet_df['target']
X = tweet_df['text']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=RANDOM_SEED)

#%% Pipeline for MultinombialNB
text_clf_mnb = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])
#text_clf_mnb = text_clf.fit(X_train,y_train)

#%% GridSearch mnb
gs_clf = GridSearchCV(text_clf_mnb,parameters,n_jobs=5)
gs_clf = gs_clf.fit(X_train,y_train)
best_score = gs_clf.best_score_
best_params = gs_clf.best_params_

#%% MultinombialNB Performance
predicted_mnb = gs_clf.predict(X_test)
pred_probs_mnb = gs_clf.predict_proba(X_test)
accuracy_score_mnb = accuracy_score(y_test,predicted) #76%
sample_size = 50
test_sample = pd.DataFrame(X_test.head(sample_size))
test_sample['probs'] = predicted_mnb[:sample_size]
test_sample['actual'] = y_test.head(sample_size)

#%% MultinombialNB Confusion Matrix
cf_mnb = confusion_matrix(y_test,pd.Series(predicted))
fig, ax = plt.subplots(figsize=(4,4)) 
sns.heatmap(cf,annot=True,fmt='g',ax=ax,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])

#%% Pipeline for SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=RANDOM_SEED)),])
text_clf = text_clf.fit(X_train,y_train)

#%% SGDClassifier Performance
predicted = text_clf.predict(X_test)
#pred_probs = text_clf.predict_proba(X_test)
accuracy_score_sgd = accuracy_score(y_test,predicted) #68%

#%% SGDClassifier Confusion Matrix
cf = confusion_matrix(y_test,pd.Series(predicted))
fig, ax = plt.subplots(figsize=(4,4)) 
sns.heatmap(cf,annot=True,fmt='g',ax=ax,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])


