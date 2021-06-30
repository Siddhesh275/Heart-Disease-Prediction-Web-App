# -*- coding: utf-8 -*-
"""
Created on Sat Mar,  30 09:08:42 2019

@author: Ashutosh Agrahari
"""

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
#import confusion_matrix
import numpy as np
import itertools
import joblib

df = pd.read_csv('heart_tidy.csv', names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']


X = df.loc[1:,features].values
Y = df.loc[1:,['target']].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.3, random_state = 0)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
joblib.dump(lr, 'Heart_model.pkl')

acc = lr.score(X_test, Y_test)
print("Accuracy: ",acc*100," %.")