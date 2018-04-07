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
        # 根据信息熵公式计算每个标签信息熵并累加到shannonEnt上
        # log(x, 10) 表示以10为底的对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#按照给定特征划分数据集
# dataSet数据集，axis是对应的要分割数据的列，value是要分割的列按哪个值分割，即找到含有该值的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 如果列标签对应的值为value，则将该条(行)数据加入到retDataSet中
        if featVec[axis] == value:
            # 取featVec的0-axis个数据，不包括axis，放到reducedFeatVec中
            reduceFeatVec = featVec[:axis]
            # 取featVec的axis+1到最后的数据，放到reducedFeatVec的后面
            reduceFeatVec.extend(featVec[axis+1:])
            # 将reducedFeatVec添加到分割后的数据集retDataSet中，此时retDataSet中没有了axis列的数据
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集当前的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历特征，即所有的列，计算每一列分割后的信息增益，找出信息增益最大的列
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
        infoGain = baseEntropy - newEntropy
        #计算最好信息收益
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 对类标签进行投票 ，找标签数目最多的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分
    # count() 方法用于统计某个元素在列表中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征是返回出现最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择一个使数据集分割后最大的特征列的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        # 子标签subLabels为labels删除bestFeat标签后剩余的标签
        subLabels = labels[:]
        splitData = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = creatTree(splitData, subLabels)
    return myTree

# myData,myLabels = creatDataSet()
# myTree = creatTree(myData,myLabels)
# print(myTree)