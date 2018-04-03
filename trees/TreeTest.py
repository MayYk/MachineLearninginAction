#!user/bin/env python
# _*_ coding:utf-8 _*_
import trees as trees
import TreePlotter as treePlt

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName)
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readline()]
lensesLabels = ['age', 'prescript', 'astimatic', 'tearRate']
lensesTree = trees.creatTree(lenses, lensesLabels)
treePlt.createPlot(lensesTree)