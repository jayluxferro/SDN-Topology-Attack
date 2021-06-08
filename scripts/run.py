#!/usr/bin/env python

import sys
import func as fx
import db

# getting inputs
host = int(sys.argv[1])
scenario = int(sys.argv[2])
params = ['iperf']
params.extend(sys.argv[3:])

iterations = db.getScenario(scenario)

for x in range(10000 - iterations):
    print("[ => ]: {}".format(iterations + x + 1))
    data = fx.iperf(params)

    # extracting the data
    tx = float(data['sum']['transfer'].split('GBytes')[0].rstrip())
    bw = float(data['sum']['bandwidth'].split('Gbits/sec')[0].rstrip())
    interval = float(data['sum']['interval'].split('-')[-1].split('sec')[0].rstrip())
    db.addData([host, tx, bw, interval, scenario])
    print('')
