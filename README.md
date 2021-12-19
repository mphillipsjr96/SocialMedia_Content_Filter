# SocialMedia_Content_Filter
Backend for a Chrome Extension that will filter social media content based on sentiment and users' choice filter words. This repository contains the models and scripts used to create the models. The extension itself can be found  <a href='https://github.com/drkinder/content-filter-chrome-extension'>here</a>.
<br>
<h2>Folders</h2>
<h3>Code</h3>
initial_cleaning.py - Script for taking the training.16000000.processed.noemoticon.csv data and turning it to bigram_tweet_df.csv
<br>initial_modeling.py - Script for the sentiment modeling
<br>preprocessing_script.py - Script for preprocessing the tweets, word filtering takes place here too
<br>topic_modeling.py - Script for topic modeling (Unfinished - not used in final extension)
<br>Capstone_Glove_Word_Embeddings.ipynb - Exploration of GloVE word vectors for classification
<br>Capstone_Fasttext_Word_Embeddings.ipynb - Exploration of fasttext word vectors for classification
<br>Tweet_Extraction_from_UMich_Tweets.ipynb - Extracting test set from random sample of tweets provided by UMSI
<br>Capstone_LinearSVC_Model_Eval.ipynb - Evaluate LinearSVC model on test set derived from UMSI tweet sample
<br>
<h3>Data</h3>
bigram_tweet_df.csv - Cut dataset containing tweets
<br>training.1600000.processed.noemoticon.csv - Cut dataset containing tokenized tweets
<br>thesaurus.json - The original thesaurus file
<br>thesaurus_lem.json - The lemmatized thesaurus file
<br>sampled_tweets.csv - Tweets sampled from UMSI tweets using VADER algorithm for preliminary sentiment labeling
<br>cleaned_human_responses.csv - Modified version of sampled_tweets with human labeling added
<br>
<h3>Models</h3>
LinearSVCModel.sav - Linear SVC Model Pickle
<br>MNBModel.sav - Multinomial Naive Bayes Model Pickle
<br>phrasemodel.sav - Phrase Model Pickle
<br>SGDModel.sav - Stochastic Gradient Descent Model Pickle
<br>
<h3>Other</h3>
Images - Confusion Matrices and Accuracy vs Model Size Graph
