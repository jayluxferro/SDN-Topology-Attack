#!/usr/bin/env python3

import db

dataStoragePath = '../results/'
data = 'data.csv'
output = 'Tx,BW,Interval,Label\n'
for x in db.getAllData():
    output += '{},{},{},{}\n'.format(x['transfer'], x['bandwidth'], x['interval'], x['scenario'])

with open(dataStoragePath + data, 'w') as f:
    f.write(output)
