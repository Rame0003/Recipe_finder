#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

def get_data(filename):
    df = pd.read_json(filename)
    new = []
    for s in df['ingredients']:
        s = ' '.join(s)
        new.append(s)
    df['ing'] = new
    return df

def get_user_ing(string, df):
    user_ing = string
    df = df.append({'ing':user_ing}, ignore_index=True)
    return df

def train_data(df, n):
    vectorizer = TfidfVectorizer(use_idf = True, smooth_idf=True, stop_words = 'english',max_features = 4000)
    ing_vect = vectorizer.fit_transform((df['ing'].values))
    vec = ing_vect.todense()
    X_train_df = vec[:-1]
    y_train_df = df['cuisine'][:-1]
    X_test_df = vec[-1]
    n = n
    model = KNeighborsClassifier(n_neighbors = n, weights='uniform', algorithm='auto', metric='minkowski')
    preds = model.fit(X_train_df,y_train_df)
    return X_test_df, model

def get_results(test, model):
    
    predicted_class = model.classes_
    predicted_single_cuisine = model.predict(test)
    predicted_cuisine = model.predict_proba(test)[0]
    match_perc,match_id = model.kneighbors(test)
    pos = np.where(predicted_class == predicted_single_cuisine)
    print ("The model predicts that the ingredients resembles %s (%f resemblence)" %(predicted_single_cuisine[0], predicted_cuisine[pos]*100))
    for i in range(len(match_id[0])):
        print ('Recipe No: %d (%f probable match)'%(match_id[0][i], match_perc[0][i]))

