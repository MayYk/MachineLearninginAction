#!user/bin/env python
# _*_ coding:utf-8 _*_
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import regTrees
import tkinter as tk
from numpy import *

# 这里引入包的顺序需要注意，matplotlib.use('TkAgg')需要放在regTrees前
# This call to matplotlib.use() has no effect because the backend has already
# been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
# or matplotlib.backends is imported for the first time.

# matplotlib中的后端 https://www.cnblogs.com/suntp/p/6519386.html

def reDraw(tolS, tolN):
    # 清空之前的图像
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    # 检查选框model tree是否被选中
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree =regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree,reDraw.testDat)
    # 绘制真实值
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(),reDraw.rawDat[:, 1].tolist(), s=5)
    # 预测值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth= 2.0)
    reDraw.canvas.draw()

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("Enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')

    try:
        tolS = 1.0
    except:
        tolS = 1.0
        print("Enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

def tkinterTest():
    root = tk.Tk()

    tk.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
    tk.Label(root, text="tolN").grid(row=1, column=0)

    tolNentry = tk.Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    tk.Label(root, text="tolS").grid(row=2, column=0)

    tolSentry = tk.Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')

    tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)

    # IntVar按钮整数值
    chkBtnVar = tk.IntVar()
    chkBtn = tk.Checkbutton(root, text="Model Tree", variable = chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)
    reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)

    tk.Button(root, text='Quit', fg="black", command=root.quit).grid(row=1, column=2)

    root.mainloop()

    # myLabel = tk.Label(root, text = 'Hello World!')
    # myLabel.grid()
    # root.mainloop()

if __name__ == '__main__':
    # tkinterTest()
    # integrationTest()
    root = tk.Tk()

    reDraw.f = Figure(figsize=(5, 4), dpi=100)  # 创建画布
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    tk.Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = tk.Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    tk.Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = tk.Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
    chkBtnVar = tk.IntVar()
    chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)
    root.mainloop()