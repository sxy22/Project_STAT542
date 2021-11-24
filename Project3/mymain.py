#!/usr/bin/env python
# coding: utf-8

import re 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_vectorizer(vocab_file):
    corpus = None
    with open(vocab_file, encoding='UTF-8') as f:
        corpus = f.readlines() 
    corpus = [w.strip() for w in corpus]
    voca = dict()
    for idx, term in enumerate(corpus):
        voca[term] = idx

    vectorizer = CountVectorizer(vocabulary=voca, ngram_range=(1, 10))
    return vectorizer

# load vocab
cleaner = re.compile('<.*?>') 
vocab_file = './myvocab.txt'
vectorizer = get_vectorizer(vocab_file)
# read data
train_file = './train.tsv'
test_file = './test.tsv'

train = pd.read_csv(train_file, sep='\t', encoding='utf-8')
train['review'] = train['review'].map(lambda s: re.sub(cleaner, '', s))

test = pd.read_csv(test_file, sep='\t', encoding='utf-8')
test['review'] = test['review'].map(lambda s: re.sub(cleaner, '', s))
pred_id = test['id']

X_train = vectorizer.transform(train['review'].values).toarray()
Y_train = train['sentiment']

X_test = vectorizer.transform(test['review'].values).toarray()

scaler = StandardScaler() 
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=vectorizer.get_feature_names())
X_test = pd.DataFrame(scaler.transform(X_test), columns=vectorizer.get_feature_names())
# fitting model
ridge = LogisticRegression(C=0.0007, random_state=2021, max_iter=1000)
ridge.fit(X_train, Y_train)
pred_test = ridge.predict_proba(X_test)
# write sub
sub = './mysubmission.txt'
mysubmission = pd.DataFrame({'id': pred_id, 'prob': pred_test[:, 1]})
mysubmission.to_csv(sub, sep='\t', index=False)

