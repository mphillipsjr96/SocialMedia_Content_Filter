# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:49:52 2021

@author: micha
"""
#%% Imports
import pandas as pd
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle

#%% Global Variables
RANDOM_SEED = 42

#%% Load Data
tweet_df = pd.read_csv('D:/Documents/Python/Data/bigram_tweet_df.csv')
y = tweet_df['target']
X = tweet_df['text']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=RANDOM_SEED)
cut = int(len(X_train)*.01)

#%% Pipeline for MultinombialNB

text_clf_mnb = Pipeline([('vect', CountVectorizer(stop_words='english',ngram_range=(1,4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])
text_clf_mnb = text_clf_mnb.fit(X_train[:cut],y_train[:cut])

#%% MultinomialNB Performance

predicted_mnb = text_clf_mnb.predict(X_test)
pred_probs_mnb = text_clf_mnb.predict_proba(X_test)
accuracy_score_mnb = accuracy_score(y_test,predicted_mnb) #76%
pred_neg = [item[0] for item in pred_probs_mnb]
sample_size = 50
test_sample = pd.DataFrame(X_test[:cut].head(sample_size))
test_sample['Negativity'] = pred_neg[:sample_size]
test_sample['preds'] = predicted_mnb[:sample_size]
test_sample['actual'] = y_test.head(sample_size)

#%% MultinombialNB Confusion Matrix
cf_mnb = confusion_matrix(y_test,pd.Series(predicted_mnb))
fig, ax = plt.subplots(figsize=(4,4)) 
sns.heatmap(cf_mnb,annot=True,fmt='g',ax=ax,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])

#%% Pipeline for SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='log',penalty='l2', alpha=1e-3, random_state=RANDOM_SEED)),])
text_clf = text_clf.fit(X_train[:cut],y_train[:cut])
predicted = text_clf.predict(X_test)
pred_probs = text_clf.predict_proba(X_test)
accuracy_score_sgd = accuracy_score(y_test,predicted)

#%% SGDClassifier Confusion Matrix
cf = confusion_matrix(y_test,pd.Series(predicted))
fig, ax = plt.subplots(figsize=(4,4)) 
sns.heatmap(cf,annot=True,fmt='g',ax=ax,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])

#%% Pipeline for LinearSVC
y = tweet_df['target']
X = tweet_df['text']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=RANDOM_SEED)

####
svc_pipe = Pipeline([('vect', CountVectorizer(stop_words='english',ngram_range=(1,3))),
                    ('tfidf', TfidfTransformer(use_idf=False)),
                    ('clf', CalibratedClassifierCV(base_estimator=LinearSVC(tol=1e-4,penalty='l2',dual=False,random_state=RANDOM_SEED),cv=5)),])
svc_clf = svc_pipe.fit(X_train[:cut],y_train[:cut])
predicted = svc_clf.predict(X_test)
pred_probs = svc_clf.predict_proba(X_test)
accuracy_score_svc = accuracy_score(y_test,predicted)

#%% LinearSVC Confusion Matrix
cf = confusion_matrix(y_test,pd.Series(predicted))
fig, ax = plt.subplots(figsize=(4,4)) 
sns.heatmap(cf,annot=True,fmt='g',ax=ax,xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])


#%% Model Performance Vis

training_sizes = [1200,12000,120000,1200000,1200,12000,120000,1200000,1200,12000,120000,1200000]
accuracies = [67.0,72.6,75.8,78.0,67.0,71.9,72.8,73.0,67.0,73.1,76.9,79.3]
pickle_sizes = [1507,13703,123937,1078013,142,757,4219,22264,988,8630,75017,613742]
models = ['MNB','MNB','MNB','MNB','SGD','SGD','SGD','SGD','SVC','SVC','SVC','SVC']

model_perf_df = pd.DataFrame()
model_perf_df['Model'] = models
model_perf_df['Training Size'] = training_sizes
model_perf_df['Accuracy'] = accuracies
model_perf_df['Pickle Size (kb)'] = pickle_sizes

import altair as alt

graph= alt.Chart(model_perf_df).mark_line().encode(
    x=alt.X('Pickle Size (kb)',scale=alt.Scale(type='log')),
    y=alt.Y('Accuracy',scale=alt.Scale(domain=[65,80])),
    color = 'Model'
    )

graph.show()




