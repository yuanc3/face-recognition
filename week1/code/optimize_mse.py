#coding:utf-8
#用框架pytorch求出loss=(5*w_1+3w_2-1)^2+(-3*w_1-4*w_2+1)^2的最小值，用梯度下降的方法.
#这个问题相当于训练一个如下的分类器
# 样本
#  特征  类别
# [5,3]   1 
# [-3,-4] -1
# [x1,x2] y_gt
# 分类器
#  y = x1*w1+x2*w2
import numpy as np
import torch

# 随机初始化模型权重
w1 = torch.randn((),requires_grad=True)
w2 = torch.randn((),requires_grad=True)
lr = 0.0005

for i in range(4000):
    loss=(5*w1+3*w2-1).pow(2)+(-3*w1-4*w2+1).pow(2)
    if i % 100 == 99:
        print("[%s] loss=%s,w1=%s,w2=%s"%(i,loss,w1.data,w2.data))
    loss.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad
        w1.grad=None
        w2.grad=None
y1 = 5*w1+3*w2
y2 = -3*w1-4*w2
print("y_pred=(%s,%s)"%(y1.data,y2.data))
