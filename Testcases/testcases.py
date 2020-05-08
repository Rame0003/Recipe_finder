#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import recipe_finder
import pytest

def data_test():
    filename = 'yummly.json'
    df = recipe_finder.get_data(filename)
    assert df is not None

def user_data():
    filename = 'yummly.json'
    df = recipe_finder.get_data(filename)
    data = 'thyme, basil'
    df1 = recipe_finder.get_user_ing(data, df)
    assert df1 is not None

def trained():
    filename = 'yummly.json'
    df = recipe_finder.get_data(filename)
    data = 'thyme, basil'
    df = recipe_finder.get_user_ing(data, df)
    test, mod = recipe_finder.train_data(df)
    assert test is not None

def out():
    filename = 'yummly.json'
    df = recipe_finder.get_data(filename)
    data = 'thyme, basil'
    df = recipe_finder.get_user_ing(data, df)
    test, mod = recipe_finder.train_data(df)
    recipe_finder.get_results(test, model, 5)
    
