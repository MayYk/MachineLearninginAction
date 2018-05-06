#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import svmMLiA as svm

# 核转换函数
def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m, 1)))
    # 如果核函数类型为'lin'
    if kTup[0] == 'lin':
        K = X * A.T
    # 如果核函数类型为'rbf':径向基核函数
    # 将每个样本向量利用核函数转为高维空间
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        # 元素间除法
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct1:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """

        :param dataMatIn:
        :param classLabels:
        :param C:
        :param toler:
        :param kTup: kTup是一个包含核信息的元组，它提供了选取的核函数的类型，比如线性'lin'或者径向基核函数'rbf'
        以及用户提供的到达率σ
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def innerLL(i, oS):
    Ei = svm.calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = svm.selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] -oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L==H')
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta.all() >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = svm.clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        svm.updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        svm.updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 在测试中使用核函数
def testRbf(k1 = 1.3):
    dataArr, labelArr = svm.loadDataSet('testSetRBF.txt')
    b, alphas = svm.smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))

    dataArr, labelArr = svm.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))

# 手写识别问题
# 图像转向量
def img2Vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImage(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2Vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImage('trainingDigits')
    b, alphas = svm.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    m,n = shape(datMat)
    errCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd] + b)
        if sign(predict) != sign(labelArr[i]):
            errCount += 1
    print('the training error rate is: %f' % (float(errCount)/m))

    dataArr, labelArr = loadImage('testDigits')
    errCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errCount += 1
    print('the test error rate is: %f' % (float(errCount)/m))

if __name__ == '__main__':
    testRbf()
    # testDigits(('rbf', 20))
