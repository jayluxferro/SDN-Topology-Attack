#!/usr/bin/python3

"""
Author: Jay
Date:   19th May, 2019
Description: iperf client parser
"""

import subprocess
import sys
import time

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

            if rcv[0] == '[SUM]' or counter == 1:
                SUM = {'interval':  rcv[-3], 'transfer':  rcv[-2], 'bandwidth':  rcv[-1]}
            else:
                data.append({'interval': rcv[-3], 'transfer': rcv[-2], 'bandwidth': rcv[-1]})
            if counter == 1:
                data.append(SUM)
            counter = counter + 1
    if counter - 2 == 0:
        counter = 3

    print({ 'data': data, 'sum': SUM, 'duration': time.time() - start, 'command': ' '.join(params), 'threads': counter - 2 })
