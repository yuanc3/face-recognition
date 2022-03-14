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
x = torch.tensor([[5,3],[-3,-4]])
y_gt = torch.tensor([1,-1])

# 随机初始化模型权重
weights = torch.randn((1,2),requires_grad=True)
lr = 0.0005

for i in range(4000):
    y_pred = (x*weights).sum(1)
    loss = (y_pred-y_gt).pow(2).sum()
    if i % 100 == 99:
        print("[%s] loss=%s,weights=%s"%(i,loss,weights.data))
    loss.backward()
    with torch.no_grad():
        weights  -= lr*weights.grad
        weights.grad=None
print("y_pred=%s"%(y_pred))
