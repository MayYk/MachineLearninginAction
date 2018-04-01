#!user/bin/env python
# _*_ coding:utf-8 _*_

from numpy import *
import matplotlib.pyplot as plt
import kNN

# group,labels = createDataSet()
# print(group)
# print(labels)
# print(classify0([0,0],group,labels,3))
# file2matrix('datingTestSet2.txt')
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
# print(datingLabels)

# Matplotlib对象简介
# FigureCanvas  画布
# Figure        图
# Axes          坐标轴(实际画图的地方)
fig = plt.figure()
#add_subplot(xyz)，将画布分为x行y列，取第z块
#add_subplot(x,y,z)，将画布分为x行y列，取第z块
ax = fig.add_subplot(111)
# 这里逗号分开两维，逗号前取行，逗号后取列
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()