#!/usr/bin/env python

import time
import subprocess
import db
import requests
import operator
from pandas import DataFrame as DF
import cm_pretty as cm_p
import numpy as np
import logger as lg
import pickle
from sklearn.metrics import  precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from math import log10 as log

results_path='../results/'
data_path = results_path + 'data/'
header=['Tx','Bw','Interval','Label']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'peru', 'teal']
figsize = (8, 4)
ratio_legend = ['80/20', '70/30', '60/40']
ratio_path_label = ['8020', '7030', '6040']
all_data_results = 'summary.csv'
#models = ['knn', 'lr', 'lsvc', 'svc', 'dt', 'rf', 'gb', 'gnb', 'mnb', 'lstm']
#modelLegend = ['KNN', 'LogisticRegression', 'LinearSVC', 'SVC', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'GaussianNB', 'MultinomialNB', 'LSTM']
models = ['knn', 'lr', 'svc', 'dt', 'rf', 'gb', 'lstm']
modelLegend = ['KNN', 'LogisticRegression', 'SVC', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'LSTM']
splitRatios = [0.2, 0.3, 0.4]
round_val = 4

def iperf(params):
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

    return { 'data': data, 'sum': SUM, 'duration': time.time() - start, 'command': ' '.join(params), 'threads': counter - 2 }

def generatePoints(length):
    return np.linspace(1, length, length)

def saveLinearModel(prefix, model):
    file_name = results_path + prefix + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        lg.success('[+] Model save: ==> {}\n'.format(prefix))

def loadLinearModel(prefix):
    model = results_path + prefix + '.pkl'
    with open(model, 'rb') as file:
        return pickle.load(file)
    return None

def rnnprint(s):
    with open(results_path + 'rnn_modelsummary.txt','w+') as f:
        print(s, f)

def cnnprint(s):
    with open(results_path + 'cnn_modelsummary.txt','w+') as f:
        print(s, f)

def plot_cm(cm, title='Confusion Matrix'):
    cmap = 'PuRd'
    cm = np.array(cm)
    cm_p.pretty_plot_confusion_matrix(DF(cm), cmap=cmap, title=title)

"""
def plotSinglePS(classifier, X_test, y_test, test_size):
    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    #disp.ax_.set_title('Precision-Recall curve: AP={0:0.2f} T={1:0.2f}'.format(disp.average_precision, test_size))
    #plt.show()
    plt.savefig(results_path + '{}_T_{}.eps'.format(disp.estimator_name, test_size))
"""
def plotNPS(model, y_test, y_pred, test_size):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    #print(precision, recall, thresholds)
    plt.figure()
    plt.step(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    avgPrecision = np.average(precision)
    legend = 'Model={0} AP={1:0.2f} T={2:0.2f}'.format(model, np.average(precision), test_size)
    plt.legend([legend])
    #plt.title('Precision-Recall curve: Model={0} AP={1:0.2f}'.format(model, np.average(precision)))
    #plt.show()
    plt.savefig(results_path + '{}_T_{}.eps'.format(model, test_size))
    plt.savefig(results_path + '{}_T_{}.png'.format(model, test_size), dpi=1200)

    # plot the log graph
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision (log10)')
    plt.legend([legend])
    plt.step(recall, [log(i) for i in precision])
    plt.savefig(results_path + '{}_T_{}_log.eps'.format(model, test_size))
    plt.savefig(results_path + '{}_T_{}_log.png'.format(model, test_size), dpi=1200)
    print(precision, recall)
    return precision, recall, legend


def plotSummary(precisionList, recallList, legends, test_size):
    plt.figure()
    counter = 0
    for p in precisionList:
        plt.step(recallList[counter], precisionList[counter], label=legends[counter])
        counter += 1
    plt.legend(legends)
    plt.xlabel('Recall')
    plt.ylabel('Precision (log10)')
    plt.savefig(results_path + 'pr_summary_{}.eps'.format(test_size))
    plt.savefig(results_path + 'pr_summary_{}.png'.format(test_size), dpi=1200)

    # log graph
    plt.figure()
    counter = 0
    for p in precisionList:
        plt.step(recallList[counter], [log(i) for i in precisionList[counter]], label=legends[counter])
        counter += 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(legends)
    plt.savefig(results_path + 'pr_summary_{}_log.eps'.format(test_size))
    plt.savefig(results_path + 'pr_summary_{}_log.png'.format(test_size), dpi=1200)


def sortData(x, y):
    L = sorted(zip(x,y), key=operator.itemgetter(0))
    new_x, new_y = zip(*L)
    return [ round(x, round_val) for x in list(new_x) ], [ round(x, round_val) for x in list(new_y) ]

def savePlotData(path, category, header, x, y, model, tsize):
    data = header
    counter = 0
    for _x in x:
        data += '{},{},{},{}\n'.format(x[counter], y[counter], model, tsize)
        counter += 1

    with open(path, 'w') as f:
        f.write(data)

def plotAllDataRecall(allData, index, models,  modelLegend):
    plt.figure(figsize=figsize)
    counter = 0
    for d in allData:
        node = d[index]
        holder = sortData(node[0], node[1])
        xticks = np.linspace(1, len(holder[0]), len(holder[0]))
        plt.plot(xticks, holder[0], '-*', color=colors[counter], label=modelLegend[counter])
        #print(node[0], node[1])
        plt.xlabel('Scenario')
        plt.ylabel('Recall')
        plt.xticks(xticks)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        ## save data
        savePlotData(data_path + 'recall/{}/{}.csv'.format(ratio_path_label[index], modelLegend[counter]), modelLegend[counter], "Scenarios, Recall, Model, TSize\n", xticks, holder[0], modelLegend[counter], ratio_legend[index])
        counter += 1
    plt.ylim((0.998, 1.00001))
    plt.title('The Recall rate of the models (' + ratio_legend[index] + ' Train-Test Split Ratio) for 20 Scenarios')
    plt.tight_layout()
    plt.savefig(results_path + 'r_model_summary_{}.eps'.format(index))
    plt.savefig(results_path + 'r_model_summary_{}.png'.format(index), dpi=1200)
    #plt.show()

def plotAllDataPrecision(allData, index, models,  modelLegend):
    plt.figure(figsize=figsize)
    counter = 0
    for d in allData:
        node = d[index]
        holder = sortData(node[0], node[1])
        xticks = np.linspace(1, len(holder[0]), len(holder[0]))
        plt.plot(xticks, holder[1], '-*', color=colors[counter], label=modelLegend[counter])
        #print(node[0], node[1])
        plt.xlabel('Scenario')
        plt.ylabel('Precision')
        plt.xticks(xticks)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        ## save data
        savePlotData(data_path + 'precision/{}/{}.csv'.format(ratio_path_label[index], modelLegend[counter]), modelLegend[counter], "Scenarios, Precision, Model, TSize\n", xticks, holder[1], modelLegend[counter], ratio_legend[index])
        counter += 1
    plt.title('The Precision of the models (' + ratio_legend[index] + ' Train-Test Split Ratio) for 20 Scenarios')
    plt.tight_layout()
    plt.savefig(results_path + 'p_model_summary_{}.eps'.format(index))
    plt.savefig(results_path + 'p_model_summary_{}.png'.format(index), dpi=1200)
    #plt.show()

def plotAllData(allData, index, modelLegend):
    plt.figure(figsize=figsize)
    counter = 0
    for d in allData:
        node = d[index]
        holder = sortData(node[0], node[1])
        plt.plot(holder[0], holder[1], '*', color=colors[counter], label=modelLegend[counter])
        #plt.axvline(x=0.86, linewidth=4, color='r')
        #print(node[0], node[1])
        counter += 1
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.gca().margins(x=0)
    plt.xlim((0.99, 1.0001))
    plt.title('Precision against Recall for ' + ratio_legend[index] + ' Train-Test Split Ratio')
    plt.tight_layout()
    plt.savefig(results_path + 'pr_summary_{}.eps'.format(index))
    plt.savefig(results_path + 'pr_summary_{}.png'.format(index), dpi=1200)
    #plt.show()

def plotAllDataAccuracy(allData, index, modelLegend):
    plt.figure(figsize=figsize)
    counter = 0
    for d in allData:
        node = d[index]
        holder = sortData(node[0], node[2])
        xticks = np.linspace(1, len(holder[0]), len(holder[0]))
        plt.plot(xticks, holder[1], '-*', color=colors[counter], label=modelLegend[counter])
        #plt.axvline(x=0.86, linewidth=4, color='r')
        #print(node[0], node[1])
        plt.xlabel('Scenarios')
        plt.ylabel('Accuracy')
        plt.xticks(xticks)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        ## save data
        savePlotData(data_path + 'accuracy/{}/{}.csv'.format(ratio_path_label[index], modelLegend[counter]), modelLegend[counter], "Scenarios, Accuracy, Model, TSize\n", xticks, holder[1], modelLegend[counter], ratio_legend[index])
        counter += 1
    plt.title('Accuracy of the models for ' + ratio_legend[index] + ' Train-Test Split Ratio')
    plt.tight_layout()
    plt.savefig(results_path + 'ac_summary_{}.eps'.format(index))
    plt.savefig(results_path + 'ac_summary_{}.png'.format(index), dpi=1200)
    #plt.show()
