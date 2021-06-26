"""
All Data Analysis
"""

import db
import func as fx

models = ['knn', 'lr', 'lsvc', 'svc', 'dt', 'rf', 'gb', 'gnb', 'mnb', 'lstm']
modelLegend = ['KNN', 'LogisticRegression', 'LinearSVC', 'SVC', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'GaussianNB', 'MultinomialNB', 'LSTM']
splitRatios = [0.2, 0.3, 0.4]
#splitRatios = [0.2]

allData = [[[[] for _ in range(3)] for _ in range(len(splitRatios))] for _ in range(len(models))] # [model] => [split] => [[recall], [precision], [accuracy]]

for d in db.fetchTable('all_data'):
    modelIndex = models.index(d['model'])
    splitIndex = splitRatios.index(d['tsize'])
    recall = d['recall']
    precision = d['precision']
    accuracy = d['accuracy']
    node = allData[modelIndex][splitIndex]
    node[0].append(recall)
    node[1].append(precision)
    node[2].append(accuracy)

dataStoragePath = fx.results_path + fx.all_data_results
_data = 'Precision,Recall,Accuracy,F1,Model,Model-Name,T-Size\n'
for ratio in splitRatios:
    for model in models:
        _res = db.fetchKPIs(ratio, model)
        _data += '{},{},{},{},{},{},{}\n'.format(_res[0], _res[1], _res[2], _res[3], model, modelLegend[models.index(model)], ratio)
# writing to data storage
with open(dataStoragePath, 'w') as f:
    f.write(_data)

for x in splitRatios:
    fx.plotAllData(allData, splitRatios.index(x), modelLegend)
    fx.plotAllDataPrecision(allData, splitRatios.index(x), models, modelLegend)
    fx.plotAllDataRecall(allData, splitRatios.index(x), models, modelLegend)
    fx.plotAllDataAccuracy(allData, splitRatios.index(x), modelLegend)
    #break

