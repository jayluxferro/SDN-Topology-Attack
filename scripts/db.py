#!/usr/bin/python

import sqlite3

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
