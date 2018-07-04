#!user/bin/env python
# _*_ coding:utf-8 _*_
from time import sleep
import re

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        # 链接相似元素项
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, '  ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def creatTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])

    freqItemSet = set(headerTable.keys())
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0:
        return None,None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            # 只对频繁项集进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # 使用排序后的频率项集对树进行填充
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    # 首先检查是否存在该节点
    if items[0] in inTree.children:
        # 存在则计数增加
        inTree.children[items[0]].inc(count)
    else:
        # 不存在则将新建该节点
        # 创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 若原来不存在该类别，更新头指针列表
        if headerTable[items[0]][1] == None:
            # 更新指向
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:#更新指向
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#节点链接指向树中该元素项的每一个实例。
# 从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾
def updateHeader(nodeToTest, targeNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targeNode

def loadSimpData():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(leafNode, prefixPath):
    # 迭代上溯整棵树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 头指针表中的元素项按照频繁度排序,从小到大
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p:str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        # 递归调用函数来创建基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])

        myCondTree, myHead = creatTree(condPattBases, minSup)
        # 将创建的条件基作为新的数据集添加到fp-tree
        # 挖掘条件FP树
        if myHead != None:
            print("conditional tree for:", newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def treeNodeTest():
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.disp()

def treeTest():
    simpData = loadSimpData()
    initSet = createInitSet(simpData)
    myFPtree, myHeaderTab = creatTree(initSet, 3)
    myFPtree.disp()
    a = findPrefixPath('x', myHeaderTab['x'][1])
    b = findPrefixPath('z', myHeaderTab['z'][1])
    c = findPrefixPath('r', myHeaderTab['r'][1])
    # print(a)
    # print(b)
    # print(c)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)

def newsTest():
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = creatTree(initSet, 100000)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print(len(myFreqList))
    print(myFreqList)

if __name__ == '__main__':
    # treeNodeTest()
    # treeTest()
    newsTest()