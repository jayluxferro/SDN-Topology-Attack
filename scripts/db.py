#!/usr/bin/python

import sqlite3
import logger as log
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def init():
    conn = sqlite3.connect('db')
    conn.row_factory = sqlite3.Row
    return conn

def addData(data):
    print(data)
    conn = init()
    cursor = conn.cursor()
    cursor.execute("insert into data(host, transfer, bandwidth, interval, scenario) values(?, ?, ?, ?, ?)", tuple(data))
    conn.commit()

def getScenario(scenario):
    conn = init()
    cursor = conn.cursor()
    cursor.execute("select count(*) from data where scenario=?", (scenario,))
    return cursor.fetchone()[0]

def getAllData():
    conn = init()
    cursor = conn.cursor()
    cursor.execute("select * from data")
    return cursor.fetchall()

def addAllData(data, tsize, y_test, y_pred, model):
    #print(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    db = init()
    cursor = db.cursor()
    cursor.execute("insert into all_data(data, tsize, precision, recall, accuracy, f1, model) values(?, ?, ?, ?, ?, ?, ?)", (data, tsize, precision, recall, accuracy, f1, model))
    db.commit()
    log.success('[+] {} <=> {} <=> {}'.format(data, tsize, model))

def getAvgPRAllData(tsize, model):
    db = init()
    cursor = db.cursor()
    cursor.execute("select avg(precision), avg(recall) from all_data where tsize=? and model=?", (tsize, model))
    return cursor.fetchone()

def fetchTable(tableName):
    db = init()
    cursor = db.cursor()
    cursor.execute('select * from {}'.format(tableName))
    return cursor.fetchall()

def fetchKPIs(ratio, model):
    db = init()
    cursor = db.cursor()
    cursor.execute('select avg(precision), avg(recall), avg(accuracy), avg(f1) from all_data where tsize=? and model=?', (ratio, model))
    return cursor.fetchone()
