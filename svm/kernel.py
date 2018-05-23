#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import svmMLiA as svm

# 核转换函数
def kernelTrans(X, A, kTup):
    """

    :param X: 数据集
    :param A: 某一行数据
    :param kTup: 核函数信息
    :return: K: 计算出的核向量
    """
    m,n = shape(X)
    K = mat(zeros((m, 1)))

    #根据键值选择相应核函数
    #lin表示的是线性核函数
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

        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C:
        :param toler:
        :param kTup: kTup是一个包含核信息的元组，它提供了选取的核函数的类型，比如线性'lin'或者径向基核函数'rbf'
        以及用户提供的到达率σ（速度参数）
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

# 主要修改所调用函数
def selectJJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 将误差矩阵每一行第一列置1，以此确定出误差不为0的样本
    oS.eCache[i] = [1, Ei]
    # 获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            # 选择有最大步长的j
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 否则，就从样本集中随机选取alphaj
        j = svm.selectJrand(i ,oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej

#更新误差矩阵
def updateEkK(oS, k):
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1, Ek]

def innerLL(i, oS):
    Ei = calcEkK(oS, i)
    # 如果标签与误差相乘之后在容错范围之外，且超过各自对应的常数值，则进行优化
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 启发式选择第二个alpha值
        j, Ej = selectJJ(i, oS, Ei)
        # 利用copy存储刚才的计算值，便于后期比较
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 保证alpha在0和C之间
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] -oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print('L==H')
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            # print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 对新的alphas[j]进行阈值处理
        oS.alphas[j] = svm.clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEkK(oS, j)
        # 如果新旧值差很小，则不做处理跳出本次循环
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print('j not moving enough')
            return 0
        # 对i进行修改，修改量相同，但是方向相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEkK(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 谁在0到C之间，就听谁的，否则就取平均值
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

# 注意这里公式的变化
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 在测试中使用核函数
def testRbf(k1 = 1.3):
    dataArr, labelArr = svm.loadDataSet('testSetRBF.txt')
    b, alphas = smoPP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
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
        s = sign(predict)
        la = sign(labelArr[i])
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))

    dataArr, labelArr = svm.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        s = sign(predict)
        la = sign(labelArr[i])
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
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
    b, alphas = smoPP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
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
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b

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

def smoPP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # 保存关键数据
    oS = optStruct1(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 迭代次数超过指定最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环
    # 选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while(iter < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        # 没有alpha更新对
        if entireSet:
            # 遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerLL(i, oS)
                # print('fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历非边界值
            for i in nonBoundIs:
                alphaPairsChanged += innerLL(i, oS)
                # print('non-bound, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果本次循环没有改变的alpha对，将entireSet置为true，
        # 下个循环仍遍历数据集
        if entireSet:
            entireSet = False
            # 如果本次循环没有改变的alpha对，将entireSet置为true，
            # 下个循环仍遍历数据集
        elif alphaPairsChanged == 0:
            entireSet = True
        # print('iteration number: %d' % iter)
    return oS.b, oS.alphas

if __name__ == '__main__':
    # testRbf(0.1)
    for i in range(1,100,10):
        print("i=%d" % (i))
        testDigits(('rbf', i))
