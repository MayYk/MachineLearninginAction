#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat):
    """

    :param dataMat:数据集
    :param topNfeat: 选择应用的N个特征
    :return:
    """
    meanVals = mean(dataMat, axis=0)
    # 去除平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # 将特征值排序
    eigValInd = argsort(eigVals)
    #从小到大对N个值排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到新空间
    lowDDatMat = meanRemoved * redEigVects
    reconMat = (lowDDatMat * redEigVects.T) + meanVals
    return lowDDatMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将所有NaN置为平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat

def pcaTest():
    dataMat = loadDataSet('testSet.txt')
    # lowDMat, reconMat = pca(dataMat, 1)
    lowDMat, reconMat = pca(dataMat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^',s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    fig.show()

def replaceTest():
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print(eigVals)

def DimenRedTest():
    n = 1000  # number of points to create
    xcord0 = [];
    ycord0 = []
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    markers = []
    colors = []
    fw = open('testSet3.txt', 'w')
    for i in range(n):
        groupNum = int(3 * random.uniform())
        [r0, r1] = random.standard_normal(2)
        if groupNum == 0:
            x = r0 + 16.0
            y = 1.0 * r1 + x
            xcord0.append(x)
            ycord0.append(y)
        elif groupNum == 1:
            x = r0 + 8.0
            y = 1.0 * r1 + x
            xcord1.append(x)
            ycord1.append(y)
        elif groupNum == 2:
            x = r0 + 0.0
            y = 1.0 * r1 + x
            xcord2.append(x)
            ycord2.append(y)
        fw.write("%f\t%f\t%d\n" % (x, y, groupNum))

    fw.close()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(xcord0, ycord0, marker='^', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    ax.scatter(xcord2, ycord2, marker='v', s=50, c='yellow')
    ax = fig.add_subplot(212)
    myDat = loadDataSet('testSet3.txt')
    lowDDat, reconDat = pca(myDat[:, 0:2], 1)
    label0Mat = lowDDat[nonzero(myDat[:, 2] == 0)[0], :2][0]  # get the items with label 0
    label1Mat = lowDDat[nonzero(myDat[:, 2] == 1)[0], :2][0]  # get the items with label 1
    label2Mat = lowDDat[nonzero(myDat[:, 2] == 2)[0], :2][0]  # get the items with label 2
    # ax.scatter(label0Mat[:,0],label0Mat[:,1], marker='^', s=90)
    # ax.scatter(label1Mat[:,0],label1Mat[:,1], marker='o', s=50,  c='red')
    # ax.scatter(label2Mat[:,0],label2Mat[:,1], marker='v', s=50,  c='yellow')
    ax.scatter(label0Mat[:, 0].tolist(), zeros(shape(label0Mat)[0]), marker='^', s=90)
    ax.scatter(label1Mat[:, 0].tolist(), zeros(shape(label1Mat)[0]), marker='o', s=50, c='red')
    ax.scatter(label2Mat[:, 0].tolist(), zeros(shape(label2Mat)[0]), marker='v', s=50, c='yellow')
    plt.show()

def pcaTestPlot():
    dataMat = replaceNanWithMean()

    # below is a quick hack copied from pca.pca()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[::-1]  # reverse
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals / total * 100

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()

if __name__ == '__main__':
    # pcaTest()
    # dataMat = replaceNanWithMean()
    # replaceTest()
    DimenRedTest()
    # pcaTestPlot()