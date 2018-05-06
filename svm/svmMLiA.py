#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *

# 相对最简单讲解
# https://www.cnblogs.com/steven-yang/p/5658362.html  及内容推荐链接

# 公式详细证明：
# https://blog.csdn.net/c406495762/article/details/78072313
# 周志华《机器学习》第六章
# https://zhuanlan.zhihu.com/p/26909540

# KKT条件：https://blog.csdn.net/johnnyconstantine/article/details/46335763

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

# aj大于H变为H，小于L变为L
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C，权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
    :param toler: 容错率
    :param maxIter: 退出前最大循环次数
    :return:
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    # 在没有任何alpha改变的情况下遍历数据集的次数
    # 该变量达到输入值maxIter时，函数结束运行并退出
    iter = 0
    while(iter < maxIter):
        # #改变的alpha对数
        alphaPairsChanged = 0
        for i in range(m):
            # 计算支持向量机算法的预测值
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            # 计算预测值与实际值的误差
            Ei = fXi - float(labelMat[i])
            # 如果不满足KKT条件（如果alpha可以更改进入优化过程）
            if ((labelMat[i] * Ei < - toler) and (alphas[i] < C)) or (labelMat[i] * Ei > toler and (alphas[i] > 0)):
                # 随机选择第二个变量alphaj
                j = selectJrand(i, m)
                # 计算第二个变量对应数据的预测值
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
                # 计算与测试与实际值的差值
                Ej = fXj - float(labelMat[j])
                # 记录alphai和alphaj的原始值，便于后续的比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 如果两个alpha对应样本的标签不相同
                if labelMat[i] != labelMat[j]:
                    # 求出相应的上下边界
                    L = max(0 , alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                # 根据公式计算未经剪辑的alphaj
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果改变后的alphaj值变化不大，跳出本次循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('j not moving enough')
                    continue
                # 否则，计算相应的alphai值
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 再分别计算两个alpha情况下对于的b值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print('iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
        # 最后判断是否有改变的alpha对，没有就进行下一次迭代
        if alphaPairsChanged == 0:
            iter += 1
        # 否则，迭代次数置0，继续循环
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas

#新建一个类的收据结构，保存当前重要的值
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = mat(zeros((self.m, 2)))

#格式化计算误差的函数，方便多次调用
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#修改选择第二个变量alphaj的方法
def selectJ(i, oS, Ei):
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
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 选择有最大步长的j
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 否则，就从样本集中随机选取alphaj
        j = selectJrand(i ,oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#更新误差矩阵
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

#SMO内循环寻找alphaj
def innerL(i, oS):
    # 计算误差
    Ei = calcEk(oS, i)
    # 违背kkt条件
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] -oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L==H')
            return 0
        eta = 2.0 * oS.X[i,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta.all() >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        # 如果有alpha对更新
        return 1
    else:
        # 否则返回0
        return 0

#SMO外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # 保存关键数据
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 迭代次数超过指定最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环
    # 选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        # 没有alpha更新对
        if entireSet:
            # 遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历非边界值
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果本次循环没有改变的alpha对，将entireSet置为true，
        # 下个循环仍遍历数据集
        if entireSet:
            entireSet = False
            # 如果本次循环没有改变的alpha对，将entireSet置为true，
            # 下个循环仍遍历数据集
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # b, alphas = smoP(dataArr,labelArr, 0.6, 0.001, 40)