#!user/bin/env python
# _*_ coding:utf-8 _*_
import matplotlib.pyplot as plt

# boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc是边框线粗细
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
# 定义决策树的叶子结点的描述属性
leafNode = dict(boxstyle='round4', fc='0.8')
# 定义决策树的箭头属性
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeText, centerPt, parentPt, nodeType):
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords ='axes fraction', xytext=centerPt, textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    # Matplotlib对象简介
    # FigureCanvas  画布
    # Figure        图
    # Axes          坐标轴(实际画图的地方)

    # 定义一个画布，背景为白色
    fig = plt.figure(1,facecolor='white')
    # 把画布清空
    fig.clf()
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    # frameon表示是否绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111,frameon = False)
    plotNode('决策节点', (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('叶节点', (0.8,0.1), (0.3,0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    # python3改变了dict.keys,返回的是dict_keys对象,支持iterable 但不支持indexable，我们可以将其明确的转化成list：
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt,nodeText):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeText)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5/plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# myTree = retrieveTree(0)
# print(myTree.keys())
# numLeaf = getNumLeafs(myTree)
# treeDepth = getTreeDepth(myTree)
# print(myTree,numLeaf,treeDepth)
# createPlot()

myTree = retrieveTree(0)
myTree['no surfacing'][3] = 'maybe'
createPlot(myTree)