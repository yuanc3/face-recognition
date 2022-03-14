#coding:utf-8
#[选做]3. 求y=x^2的最小值，用梯度下降的方法，不使用框架
# 求y=x^2的最小值
# dy/dx = 2*x 

# 使用梯度下降法
# 初始化
lr=0.1
x=5
y=x*x
print("y=%s,x=%s"%(y,x))
for i in range(100):
# if y-y_last<0.0000001
    #x = x - lr * dy/dx
    x = x - lr * 2*x
    y=x*x
    print("y=%s,x=%s"%(y,x))
