#!user/bin/env python
# _*_ coding:utf-8 _*_
from math import log
import operator

def creatDataSet():
    dataSet = [[1,1,'Yes'], [1,1,'Yes'], [1,0,'No'], [0,1,'No'], [0,1,'No']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # log(x, 10) 表示以10为底的对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        print('------------------------------------------------3')
        infoGain = baseEntropy - newEntropy
        #计算最好信息收益
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]

def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征是返回出现最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myData,myLabels = creatDataSet()
myTree = creatTree(myData,myLabels)
print(myTree)