#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt

def treeNode():
    def __init__(self, feat,val,right,left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:]
    return mat0, mat1

# 生成叶节点，目标变量特征的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

# 在给定数据上计算目标变量的平方误差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# 寻找数据的最佳二元切分方式
def chooseBestSplit(dataSet, leafType = regLeaf,errType = regErr,ops = (1,4)):
    # tolS:容许的误差下降值； tolN:切分的最少样本数
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有值都相等则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestVal = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

# glo_mean = 0
def getMean(tree):
    # global glo_mean
    # glo_mean += 1
    # print(glo_mean)
    # print(tree)
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return  (tree['left'] + tree['right'])/2.0

# glo_prune = 0
def prune(tree, testData):
    """

    :param tree: 待剪枝的树
    :param testData: 剪枝所需测试数据
    :return:
    """
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 假设发生过拟合，采用测试数据对树进行剪枝
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet  = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'],2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean,2))
        if errorMerge < errorNoMerge:
            # global glo_prune
            # glo_prune += 1
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

# 将数据集格式化成目标变量Y和自变量X
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones(m,n))
    Y = mat(ones(m,1))
    # X与Y数据格式化
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
    ws = xTx * (X.T * Y)
    return ws, X, Y

# 当数据不再需要切分的时候生成叶节点的模型
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

# 给定数据集上计算误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

def cart():
    myDat = loadDataSet('ex00.txt')
    myMat = mat(myDat)
    createTree(myMat)

def plotCart():
    myDat1 = loadDataSet('ex0.txt')
    # myDat = loadDataSet('ex00.txt')
    myMat1 = mat(myDat1)
    tree1 = createTree(myMat1)
    plt.plot(myMat1[:,1],myMat1[:,2],'ro')
    plt.show()

    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    tree2 = createTree(myMat2)
    myTree = createTree(myMat2, ops=(0,1))
    plt.plot(myMat2[:, 0], myMat2[:, 1], 'ro')
    plt.show()

    myDatTest = loadDataSet('ex2Test.txt')
    myMat2Test = mat(myDatTest)
    tree3 = prune(myTree, myMat2Test)
    plt.plot(myMat2Test[:, 0], myMat2Test[:, 1], 'ro')
    plt.show()

def plotLine():
    myDat = loadDataSet('ex0.txt')
    myMat = mat(myDat)
    myTree = createTree(myMat, ops = (0,1))
    prune(myTree, myMat)
    plt.plot(myMat[:,0], myMat[:,1],'ro')
    plt.show()

if __name__ == '__main__':
    # cart()
    plotCart()
    # plotLine()
