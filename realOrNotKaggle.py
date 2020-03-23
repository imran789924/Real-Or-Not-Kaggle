#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:35:14 2020

@author: imran
"""


def process_X(X, size):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    import re
    
    corpus = []
    for i in range(0, size):
        review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=X['text'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    
    return X




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('/home/imran/PROGRAMS/Machine Learning/Real or Not Kaggle/train.csv')
X = dataset.iloc[:, 3:4]
y = dataset.iloc[:, 4]

#X_test_raw = pd.read_csv('/home/imran/PROGRAMS/Machine Learning/Real or Not Kaggle/test.csv')
#X_test = X_test_raw.iloc[:, 3:4]


X = process_X(X, 7613)
#X_test = process_X(X_test, 3263)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
#classifier.fit(X_train, y_train)
classifier.fit(X_train, y_train)


y_pred =classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
output = pd.DataFrame({'id': X_test_raw.id, 'target': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
'''