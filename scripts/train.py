#!/usr/bin/env python3

"""
Author: Jay Lux Ferro
Date:   14th June, 2021
Task:   Training models
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import func as fx
import rnn_models as rmd
import sys
import pprint
import logger as lg
import math
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pickle
import os
import db
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import *

dataSourcePath = '../results/'

def runScenario(data_file, test_size):
    # data source
    data_file = '{}.csv'.format(data_file)
    df = pd.read_csv(data_file, delimiter=',')
    features_columns = []
    for x in range(len(fx.header) - 1):
        features_columns.append(x)
    features = df.iloc[:, features_columns].values
    target = df.iloc[:,[len(fx.header) - 1]].values
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, test_size=test_size)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # testing knn classifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lg.success('KNN: {:.4f}'.format(model.score(X_test, y_test)))
    cm = confusion_matrix(y_test, y_pred)
    fx.saveLinearModel('knn', model)
    db.addAllData(data_file, test_size, y_test, y_pred, 'knn')

    # testing logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lg.success('Logistic Regression: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('lr', model)
    db.addAllData(data_file, test_size, y_test, y_pred, 'lr')

    # testing linear svc
    model = LinearSVC()
    model.fit(X_train, y_train)
    lg.success('LinearSVC: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('lsvc', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'lsvc')

    # testing svc
    model = SVC()
    model.fit(X_train, y_train)
    lg.success('SVC: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('svc', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'svc')

    # testing decision tree
    model = DecisionTreeClassifier(random_state=0, max_depth=7)
    model.fit(X_train, y_train)
    lg.success('Decision Tree: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('dt', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'dt')

    # testing random forest
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    lg.success('Random Forest: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('rf', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'rf')

    # testing gradientboosting classifier
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    lg.success('Gradient Boosting: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('gb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'gb')

    # testing naive bayesian classifiers
    model = GaussianNB()
    model.fit(X_train, y_train)
    lg.success('Gaussian NB: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('gnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'gnb')

    model = BernoulliNB()
    model.fit(X_train, y_train)
    lg.success('Bernoulli NB: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('bnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'bnb')

    model = MultinomialNB()
    model.fit(X_train, y_train)
    lg.success('Multinomial NB: {:.4f}'.format(model.score(X_test, y_test)))
    fx.saveLinearModel('mnb', model)
    y_pred = model.predict(X_test)
    db.addAllData(data_file, test_size, y_test, y_pred, 'mnb')

    # neural network
    batch_size=64
    epochs = 10
    lg.warning('\n\nNeural Network')
    model = rmd.lstm(len(X_train[0]))
    try:
        plot_model(model, to_file=dataSourcePath + './rnn_lstm.eps')
    except:
        pass
    plot_model(model, to_file=dataSourcePath + './rnn_lstm.png')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    res = model.evaluate(X_test, y_test)
    lg.success('LSTM Accuracy: {:.4f}'.format(res[3]))
    y_pred = np.around(model.predict(X_test))
    y_pred = np.array([[int(i)] for i in y_pred])
    model.save(dataSourcePath + 'lstm_model.h5')
    lg.success('[+] Model saved')
    db.addAllData(data_file, test_size, y_test, y_pred, 'lstm')


# running a set of scenarios
data_source = ['data']
splits = [0.2, 0.3, 0.4]
iterations = 20
for _ in range(iterations):
    for data in data_source:
        for split in splits:
            runScenario(dataSourcePath + data, split)
            sys.exit()
