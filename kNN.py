#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import operator

#初始数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#分类器
#inx表示输入向量，dataSet表示输入的训练样本集，labels是标签向量，参数k表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    # dataSet.shap表示数组各维的大小
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

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))


group,labels = createDataSet()
print(group)
print(labels)
print(classify0([0,0],group,labels,3))