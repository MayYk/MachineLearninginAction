#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#初始数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#分类器
#inx表示输入向量，dataSet表示输入的训练样本集，labels是标签向量，参数k表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    # dataSet.shape[x]表示数组各维的大小,x=1为第一维的长度，x=2为第二维的长度。
    dataSetSize = dataSet.shape[0]

    # tile(inX,(a,b))函数将inX重复a行，重复b列
    diffMat = tile(inX,(dataSetSize, 1)) - dataSet

    #返回diffMat的2次方
    sqDiffMat = diffMat ** 2

    #axis=1表示按行相加 , axis=0表示按列相加
    sqDistance = sqDiffMat.sum(axis=1)

    distances = sqDistance ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #计算每个类别的样本数。字典get()函数返回指定键的值，如果值不在字典中返回默认值0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #reverse降序排列字典
    #python2版本中的iteritems()换成python3的items()
    #key=operator.itemgetter(1)按照字典的值(value)进行排序
    #key=operator.itemgetter(0)按照字典的键(key)进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse= True)

    return sortedClassCount[0][0]

# 文本 to 矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)

    #zeros((a,b))生成一个a行b列的0矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        #str.strip('0')，移除字符串首尾指定字符'0'
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = dataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is: %d" %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']

    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles =  float(input('frequent filer miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))

    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])

    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print('You will probaly like this person:',resultList[classifierResult - 1])

# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# datingDataMat = autoNorm(datingDataMat)[0]
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()
datingClassTest()
# classifyPerson()