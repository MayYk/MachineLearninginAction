#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 矩阵求导：https://zhuanlan.zhihu.com/p/24709748

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

# 计算最佳拟合直线
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # np.linalg.det()：矩阵求行列式（标量）
    # np.linalg.inv()：矩阵求逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do i inverse")
        return
    # .I求逆矩阵
    # ws = linalg.solvs, xMat.T * yMat)
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵
    weights = mat(eye((m)))
    for j in range(m):
        # 权重值大小以指数及衰减
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotStandRegres(xArr,yArr):
    ws = standRegres(xArr, yArr)
    print(ws)

    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flatten：用于array和mat的降维
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

def plotLwlr(xArr, yArr):
    yHat_0 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_1 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.003)

    xMat = mat(xArr)
    yMat = mat(yArr)
    # 排序，返回索引值
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    # 中文显示的bug
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_0[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title('k=1.0')
    axs1_title_text = axs[1].set_title('k=0.01')
    axs2_title_text = axs[2].set_title('k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

def rssError(yArr, yHatArr):
    su = ((yArr - yHatArr)**2).sum()
    print(su)
    return su

def agePredict1():
    abX,abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    rssError(abY[0:99], yHat01.T)
    rssError(abY[0:99], yHat1.T)
    rssError(abY[0:99], yHat10.T)

def agePredict2():
    abX,abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    rssError(abY[100:199], yHat01.T)
    rssError(abY[100:199], yHat1.T)
    rssError(abY[100:199], yHat10.T)

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 岭回归
# 计算回归系数
def ridgeRegres(xMat, yMat, lam =0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
# 在一组λ上测试结果
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # mean 求平均
    yMean = mean(yMat, 0)
    # 数据标准化
    yMat = yMat - yMean
    xMat = regularize(xMat)

    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat
# 岭回归画图
def ridgeTestPlot():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

# 前向逐步回归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """

    :param xArr:
    :param yArr:
    :param eps:     每次迭代需要调整的步长
    :param numIt:   优化过程需要迭代的次数
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def plotStageWise():
    abX, abY = loadDataSet('abalone.txt')
    returnMat = stageWise(abX, abY,0.001,5000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    plt.show()

# 书中代码api已被弃置，从网上找到原有数据集
# http://code.google.com/apis/shopping/search/v1/getting_started.html
from time import sleep
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    # 防止短时间过多API调用，许多网站还有防爬虫机制
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/prodects?key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr, setNum)

    # TODO python3改版，原有方法存在问题，原有api被弃置，暂时未验证
    # 这里url编码可能存在问题，留待后续学习爬虫时解决
    pg = urlopen(searchURL)
    retDict = json.load(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" % i)

# 网上搬的代码，进行页面的数据解析
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-03
    """
    # 打开并读取HTML文件
    opFile = 'lego/lego%s.html' % inFile
    with open(opFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)

    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)

    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = "%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            # print("商品 #%d 没有出售" % i)
            pass
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                # print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = "%d" % i)

def buildModel(lgX,lgY):
    m,n = shape(lgX)
    lgX1 = mat(ones(m, n+1))
    lgX1[:,1:5] = mat(lgX)

    ws = standRegres(lgX1,lgX)
    s1 = lgX1[0] * ws
    s2 = lgX1[-1] * ws
    s3 = lgX1[43] * ws

def crossValidation(xArr, yArr, numVal=10):
    m =len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        # 将序列所有元素随机排序
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain)/varTrain
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]

    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term:", -1 * sum(multiply(meanX, unReg)) + mean(yMat))


def legoDataTest():
    lgX = []
    lgY = []
    scrapePage(lgX,lgY,8288,2006,800,49.99)
    scrapePage(lgX,lgY,10030,2002,3096,269.99)
    scrapePage(lgX,lgY,10179,2007,5195,499.99)
    scrapePage(lgX,lgY,10181,2007,3428,199.99)
    scrapePage(lgX,lgY,10189,2008,5922,299.99)
    scrapePage(lgX,lgY,10196,2009,3263,249.99)
    print("----------------")
    print(lgX)
    print(lgY)
    print("----------------")
    crossValidation(lgX, lgY, 10)

if __name__ == '__main__':
    # xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[0:2])
    # plotStandRegres(xArr,yArr)

    # lwlr(xArr[0], xArr, yArr, 1.0)
    # lwlr(xArr[0], xArr, yArr, 0.001)
    # plotLwlr(xArr,yArr)
    # agePredict1()
    # agePredict2()
    # ridgeTestPlot()
    # plotStageWise()
    legoDataTest()