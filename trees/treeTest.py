#!user/bin/env python
# _*_ coding:utf-8 _*_
import trees as trees
import treePlotter as treePlt

# 决策树分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # 若为字典，递归的寻找testVec
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 决策树的序列化
def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName)
    pickle.dump(inputTree, fw)
    fw.close()

# 读取序列化的树
def grabTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)
if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.creatTree(lenses, lensesLabels)
    treePlt.createPlot(lensesTree)
