#!/usr/bin/python3

"""
Author: Jay
Date:   19th May, 2019
Description: iperf client parser
"""

import subprocess
import sys
import time
import requests
import json

# defaults
url = 'http://127.0.0.1:8000/'
clientIP = '127.0.0.1'


def sendData(d):
    # send ping
    rtt = subprocess.check_output(['ping', '-c', '1', clientIP], shell=False).split()[-2].split('/')[0]
    sendPing({'duration': str(rtt)})
    
    print(d)
    session = requests.session()
    res = requests.post(url + 'iperf/', data=d)
    print(res.text)


def sendPing(d):
    print(d)
    session = requests.session()
    res = requests.post(url + 'ping/', data=d)
    print(res.text)

if __name__ == "__main__":
    params = ['iperf']
    params.extend(sys.argv[1:])
    start = time.time()
    res = subprocess.check_output(params, shell=False).decode().split('\n')
   
    # removing last empty field
    res.pop()

    init = False 
    
    SUM = {}
    data = []
    counter = 1
    for x in res:
        if x.lower().find('transfer') != -1:
            init = True

        if init == True and x.lower().find('transfer') == -1 and x != '':
            rcv = x.split('  ')

            rcv[-3] = rcv[-3].split(' ')[0].split('-')[-1]
            rcv[-2] = rcv[-2].split(' ')[0]
            rcv[-1] = rcv[-1].split(' ')[0]
            if rcv[0] == '[SUM]' or counter == 1:
                SUM = {'interval':  rcv[-3], 'transfer':  rcv[-2], 'bandwidth':  rcv[-1]}
            else:
                data.append({'interval': rcv[-3], 'transfer': rcv[-2], 'bandwidth': rcv[-1]})
            if counter == 1:
                data.append(SUM)
            counter = counter + 1
    if counter - 2 == 0:
        counter = 3

    fData = { 'data': str(data), 'duration': str(time.time() - start), 'command': ' '.join(params), 'threads': counter - 2, 'start': str(start), 'stop': str(time.time()), 'interval': SUM['interval'], 'transfer': SUM['transfer'], 'bandwidth': SUM['bandwidth']}
    sendData(fData)
