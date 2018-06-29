#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt
import urllib
import json
from time import sleep

# chinacity：中国所有城市列表及经纬度---------原始数据

# citytest：程序所选择的一些城市
# chinaplace: 所选择城市经百度地图API获得经纬度

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        # 用第j个特征最大值减去最小值得出特征值范围
        rangeJ = float(max(dataSet[:, j]) - minJ)
        #创建簇矩阵的第J列,random.rand(k,1)表示产生(10,1)维的矩阵，其中每行值都为0-1中的随机值
        #可以这样理解,每个centroid矩阵每列的值都在数据集对应特征的范围内,那么k个簇中心自然也都在数据集范围内
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-均值算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # 初始化最小距离最正无穷；最小距离对应索引为-1
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分K-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0,).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        # SSE:Sum of Squared Error 无差平方和
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit:", sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("The bestCentToSplit is:", bestCentToSplit)
        print("The len of bestClustAss is:", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

# Yahoo API被弃置，改为百度地图API
def geoGrabBaidu(stAddress, city):
    apiStem = 'http://api.map.baidu.com/geocoder/v2/?'
    params = {}
    params['address'] = stAddress
    params['city'] = city
    params['ret_coordtype'] = 'wgs84'
    params['ak'] = 'slQgcgpsRGbTRE7orGCeZKRlSgRvROZC'
    params['output'] = 'json'

    url_params = urllib.parse.urlencode(params)
    baiduApi = apiStem + url_params
    # print(baiduApi)
    c = urllib.request.urlopen(baiduApi)
    return json.loads(c.read())

# 将获取的信息封装保存到文件
def massPlaceFind(fileName):
    fw = open('chinaplace.txt','w', encoding="utf-8")
    for line in open(fileName, encoding="utf-8").readlines():
        line = line.strip()
        lineArr = line[0:2]
        retDict = geoGrabBaidu(line,lineArr)
        if retDict['status'] == 0:
            lat = float(retDict['result']['location']['lat'])
            lng = float(retDict['result']['location']['lng'])
            fw.write("%s\t%f\t%f\n" % (line, lat, lng))
            print("%s\t%f\t%f" % (line, lat, lng))
        else:
            print("error featching")
        # sleep改为2，这里为1的话，调取api总是超时，不知道是不是百度频率限制
        sleep(2)
    fw.close()

# 返回地球表面两点之间的距离
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi/180) * sin(vecB[0, 1] * pi/180)
    b = cos(vecA[0, 1] * pi/180) * cos(vecB[0, 1] * pi/180) * cos(pi * (vecB[0, 0]- vecA[0, 0])/180)
    return arccos(a + b) * 6371.0

# 将文本文件中的城市及进行聚类并画出结果
def clusterClubs(numClust=5):
    datList = []
    for line in open('chinaplace.txt',  encoding="utf-8").readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[2]), float(lineArr[1])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust,distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
    # imgP = plt.imread('Portland.png')
    # ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon = False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)

    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


def centTest():
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    min0 = min(dataMat[:, 0])
    min1 = min(dataMat[:, 1])
    max1 = max(dataMat[:, 1])
    max0 = max(dataMat[:, 0])
    print(min0)
    print(min1)
    print(max0)
    print(max1)

def kMeanTest():
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],c='red')
    centroids, clusterAssment = kMeans(dataMat,5)
    # print(centroids)
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], c='blue')
    plt.show()

def bikMeanTest():
    datMat3 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat3, 3)

    xArr = datMat3[:, 0].flatten().A[0]
    yArr = datMat3[:, 1].flatten().A[0]
    xArr1 = centList[:, 0].flatten().A[0]
    yArr1 = centList[:, 1].flatten().A[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr, yArr, c='blue')
    ax.scatter(xArr1, yArr1, c='red')
    plt.show()
    # print(centList)

if __name__ == '__main__':
    # centTest()
    # kMeanTest()
    # bikMeanTest()
    # massPlaceFind('citytest.txt')
    # ss = geoGrabBaidu("山东莘县", '山东')
    # print('-------')
    clusterClubs(3)

