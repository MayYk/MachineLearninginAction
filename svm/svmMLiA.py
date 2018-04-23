#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *

# 获取文件数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i, m):
    """

    :param i:alpha的下标
    :param m:alpha的数目
    :return:
    """
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    pass