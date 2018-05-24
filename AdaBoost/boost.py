#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.1],
                      [2., 1.]])

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def createPlot(dataMat, classLabels):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            xcord1.append(dataMat[i, 0]), ycord1.append(dataMat[i, 1])
        else:
            xcord0.append(dataMat[i, 0]), ycord0.append(dataMat[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('decision stump test data')
    plt.show()

    # fig = plt.figure()
    # fig.clf()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].tolist(), dataMat[:,1].tolist())
    # plt.show()

# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)

    numSteps = 10.0
    # bestStump字典用于存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    # minError初始为无穷大
    minError = inf
    # 数据集所有特征遍历
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst



if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    # createPlot(dataMat, classLabels)
    D = mat(ones((5, 1)) / 5)
    buildStump(dataMat, classLabels, D)

