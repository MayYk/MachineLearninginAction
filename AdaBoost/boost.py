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
    ax.scatter(xcord1, ycord1, marker='o', s=90, c='red')
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
        # 如果小于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 在一个加权数据集中循环，并找到具有最低错误率的单层决策树
# 遍历stumpClassify()函数所有的可能输入值，找到数据集上最佳的单层决策树
# 单层决策树生成
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
    # 最外层循环为遍历特征，次外层循环为遍历的步长，最内层为是否大于或小于阀值
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            # lt:less than，gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)
                # 构建errArr列向量，默认所有元素都为1，如果预测的结果和labelMat中标签值相等，则置为0
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 将errArr向量和权重向量D的相应元素相乘并求和，得到数值weightedError
                # AdaBoost和分类器交互的地方
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))

                # 将当前的错误率与已有的最小错误率进行对比，如果当前的值较小，那么就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainsDS(dataArr, classLabels, numIt = 40):
    """

    :param dataArr:
    :param classLabels:
    :param numIt:       迭代次数
    :return:
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # D为概率分布向量，所有元素之和为1，代表每个数据点的权重，所以都初始化为 1/m
    D = mat(ones((m, 1)) / m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        # 确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: ', classEst.T)
        # 计算下一次迭代中的新的权重向量D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        # 错误率累加计算s
        aggClassEst += alpha * classEst
        print('aggClassEst:', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print('total error:', errorRate, '\n')
        if errorRate == 0.0:
            break
    # return weakClassArr
    return weakClassArr, aggClassEst

# 基于AdaBoost的分类
# 一个或者多个待分类样例datToClass
# classifierArr：多个弱分类器组成的数组
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is: ', ySum * xStep)

if __name__ == '__main__':
    # dataMat,classLabels = loadSimpData()
    # createPlot(dataMat, classLabels)
    # D = mat(ones((5, 1)) / 5)
    # buildStump(dataMat, classLabels, D)
    # DS：decision stump（单层决策树）
    # adaBoostTrainsDS(dataMat, classLabels, 9)

    # classifierArr = adaBoostTrainsDS(dataMat, classLabels, 30)
    # print(classifierArr)
    # lab = adaClassify([0, 0], classifierArr)
    # lab2 = adaClassify([[5, 5], [0, 0]], classifierArr)
    # print(lab2)

    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainsDS(dataArr, labelArr, 10)

    classifierArray, aggClassEst = adaBoostTrainsDS(dataArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)

    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # prediction10 = adaClassify(testArr, classifierArray)
    #
    # errArr = mat(ones((67, 1)))
    # errArr[prediction10 != mat(testLabelArr).T].sum()

