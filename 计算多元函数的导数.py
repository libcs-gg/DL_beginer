# -*- coding: UTF-8 -*-
'''
    求形如 Y=f(X,W)= g(X*W) 的多元复合函数的导数(近似值)，此类函数为深度学习神经网络计算中常用的函数。
    并用matplotlib作复合函数与导数的图像
'''
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from typing import Callable
from typing import List


##  第I部分 求实函数与n元内积函数的复合函数的导数，其中自变量是n维数组X（对应于神经网络之特征）与W（权重）

#I.1  定义求导 方法： deriv()
## 定义一元实函数的求导方法（FF是函数，XX是实变量区间）：
def deriv(F: Callable[[np.ndarray], np.ndarray], X: np.ndarray, delta: float = 0.001 ):
    return ((F(X+delta)-F(X-delta))/(2*delta))

#I.2  定义对矩阵自变量实函数的求导方法：grad_01()
# 首先定义函数组
Array_Function = Callable[[np.ndarray], np.ndarray]
#定义计算矩阵函数f(X,W)=Sigma(dot(X，W))对于第一个参数X的导数，此时视W为常量！
def grad_01(X:np.ndarray, W:np.ndarray,
           Sigma: Array_Function)-> np.ndarray:
    '计算矩阵函数f(X,W)=Sigma(dot(X，W))对于第一个自变量X的导数，其中X的列数必须等于W的行数 '
    assert X.shape[1] == W.shape[0]  #这是矩阵X与W能够相乘的条件
    N = np.dot(X, W)
    S = Sigma(N)  #函数值
    # 计算dS/dN
    dSdN = deriv(Sigma, N)
    dNdX = np.transpose(W, (1, 0))
    y = np.dot(dSdN, dNdX)
    return y

##  定义6个函数
def id(x:np.ndarray) -> np.ndarray:
    '''恒同函数'''
    return x

def square(x:np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.power(x, 2)

def leaky_relu(x:np.ndarray)->np.ndarray:
    ''':relu函数用于ndarray的每个元素'''
    return np.maximum(0.2*x, x)

def sigmoid(x:np.ndarray)->np.ndarray:
    '''sigmoid函数用于ndarray的每个元素'''
    return 1/(1+np.exp(-x))

def sin(x:np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.sin(x)

def cos(x:np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.cos(x)

def d_relu(x:np.ndarray)->np.ndarray:
    ''':relu的导数函数用于ndarray的每个元素'''
    if x < 0:
        y = 0.2
    else:
        y = 1
    return y

def d_sigmo(x:np.ndarray)->np.ndarray:
    '''sigmoid的导数函数用于ndarray的每个元素'''
    y = np.exp(-x)*(1/(1+np.exp(-x))**2)
    return y

#I.3  计算的例子
X = np.linspace(0, 4, 16).reshape(4, 4)
np.random.seed(0)
W1 = np.random.rand(4)
W = W1.reshape(4, 1)
print('自变量 X=(xij) =', X,'其中 i,j=1,2,3,4;\n 权重参数 W=(wj)', W, '\n','矩阵相乘 X*W=', np.dot(X, W))
print('-----------例 1-------------')
Y = grad_01(X, W, sin)
print('计算结果为：Y=df/dx=', Y)
print('其中：f(x,w)=sin(X*W)，df/dx=(df/dxij), i,j=1,2,3,4  ---------------\n')
print(' 验算：')
z11=cos(np.dot(X[0], W))*W[0]
z23=cos(np.dot(X[1], W))*W[2]
z44=cos(np.dot(X[3], W))*W[3]
print('z11=', z11, '\n', 'z23=', z23, '\n', 'z44=', z44 )

print('-------------例 2-------------')
Y = grad_01(X, W, leaky_relu)
print('计算结果为：Y=df/dx=', Y)
print('其中：f(x,w)=leaky_relu(X*W)，df/dx=(df/dxij), i,j=1,2,3,4  ---------------\n')
print(' 验算：')
z11=d_relu(np.dot(X[0], W))*W[0]
z23=d_relu(np.dot(X[1], W))*W[2]
z44=d_relu(np.dot(X[3], W))*W[3]
print('z11=', z11, '\n', 'z23=', z23, '\n', 'z44=', z44 )

print('-------------例 3-------------')
Y = grad_01(X, W, sigmoid)
print('计算结果为：Y=df/dx=', Y)
print('其中：f(x,w)=sigmoid(X*W)，df/dx=(df/dxij), i,j=1,2,3,4  ---------------\n')
print(' 验算：')
z11=d_sigmo(np.dot(X[0], W))*W[0]
z23=d_sigmo(np.dot(X[1], W))*W[2]
z44=d_sigmo(np.dot(X[3], W))*W[3]
print('z11=', z11, '\n', 'z23=', z23, '\n', 'z44=', z44 )


input('第一部分结束。下面是第二部分：ok ？')

##  第II部分 自变量X（对应于神经网络之特征）与W（权重）为矩阵的情况下，计算“梯度”

#II.1 定义计算 矩阵函数 L = L(S) = L(g(N)), 其中N=np.dot(X，W) 对于 xij 的导数，此时视W为常量！
'''
相关的数学描述：
自变量矩阵X型为（m,k） X=(xij)，其中 i=1,...,m; j=1,...,k，下面记Xi为X的第i行；
权重参数矩阵型为（k,n） W=(wij),其中 i=1,...,k; j=1,...,n，下面记Wi为W的第i行，Wj为W的第j列，；
矩阵相乘得中间变量N型为（m,n）N=np.dot(X, W) ,这里是矩阵相乘，N表示X的加权平均值。
对N进行“池化”；S=sigma(N),池化函数sigma(t)是一个实函数，作用到矩阵的每个元素，结果是与N同型的矩阵，型为（m,n）。
函数L对S的各元素求和：L=L(S)=s11+...+smn

我们要计算矩阵函数L=L(S)=L(S(N(X，W)))对于第一个自变量X的偏导数，即L对于每个xij的偏导数。结果是一个与X同型的矩阵。
变量关系图如下：
X1,...,Xm -> N -> S -> L

根据复合函数求导法则，
DL/DXi = DL/DS(S(N(X，W))) * DS/DN(N(X，W)) * DN/DXi(X,W)
计算： 
1、DN/DXi(X,W)=(DN/Dxi1,DN/Dxi2,...,DN/Dxik),而
DN/Dxi1=w11,...,DN/Dxik=w1k, 所以这是W1的转置：DN/DXi(X,W)=np.transpose(W1)，其中W1为W的第1行；
...
同理，DN/Dxik(X,W)=np.transpose(Wk)，其中Wk为W的第k行。
综合上述，DN/DXi(X,W)=np.transpose(W, (1, 0))，这是W的转置。
2、设sigma(N)=F(N),DS/DN(N(X，W))=（DF/DNij）
3、根据上述定义，L(S) 对元素 sij 的导数均为1，故可表示成与S同型的矩阵，元素均为1.
    DL/DS = np.ones_like(S) 。
4、DL/DXi = np.ones_like(S) * DS/DN(N(X，W)) * np.transpose(W, (1, 0))
'''
def grad_02(X:np.ndarray, W:np.ndarray,
           Sigma: Array_Function)-> np.ndarray:
    '计算矩阵函数L=L(S)=L(g(N)),其中N=np.dot(X，W)对于第一个自变量X的导数，其中X的列数必须等于W的行数 '
    assert X.shape[1] == W.shape[0]  #这是矩阵X与W能够相乘的条件，下面的np.dot()是矩阵乘法，结果是矩阵
    N = np.dot(X, W)
    S = Sigma(N)  # 函数值
    # L(S) 将S的所有元素相加：L(S)=s11+s12+s13+...+s32, S=(sij)
    L = np.sum(S)
    #dLdS 根据上述定义，L(S) 对sij 的导数均为1 ！可表示成与S同型的矩阵，元素均为1.
    dLdS = np.ones_like(S)
    # 计算dS/dN 结果是与N同型的矩阵，注意：N的行数与X同，列数与W同。
    dSdN = deriv(Sigma, N)
    # 计算dLdN，结果是矩阵，行数与S同即与N同，也就是与X的行数相同，列数与W相同
    dLdN = dLdS*dSdN
    # dNdX 矩阵，行数=W的列数，列数=W的行数
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dLdN, dNdX)
    y = dLdX
    return y


# II.2 计算的例子
print('-------------例 4-------------')
np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)
Sigma = sigmoid
L = np.sum(Sigma(np.dot(X, W)))
print('X=:', X, '\n')
print('L=', L)
print('------------- 下面是 函数L 关于各自变量 xij 的导数: -------------')
DL = grad_02(X, W, Sigma)
print(DL)

## 验算：把x11增加0.01，计算L的增量比，与DL的（1,1）项作比较
X1 = X.copy()
X1[0, 0] += 0.001 # X中把x11增加0.01
delta_L = np.sum(Sigma(np.dot(X1, W)))-np.sum(Sigma(np.dot(X, W)))

print('------------- 下面是 函数L 关于自变量 x11 的增量比: x11的增量为 0.001 -------------')
# L的增量比
test11 = delta_L/0.001
print(test11)

#II.3 定义计算 矩阵函数 L = L(S) = L(g(N)), 其中N=np.dot(X，W) 对于 wij 的导数，此时视X为常量！
'''
数学描述：
矩阵X型为（m,k） X=(xij)，其中 i=1,...,m; j=1,...,k，下面记Xi为X的第i行；
权重参数矩阵型为（k,n） W=(wij),其中 i=1,...,k; j=1,...,n，下面记Wi为W的第i行，Wj为W的第j列，W是自变量；
矩阵相乘得中间变量N型为（m,n）N=np.dot(X, W) ,这里是矩阵相乘，N表示X的加权平均值。
我们要计算矩阵函数L=L(S)=L(S(N(X，W)))对于第2个自变量W的偏导数，即L对于每个wij的偏导数。结果是一个与W同型的矩阵。
变量关系图如下：
W1,...,Wk -> N -> S -> L

根据复合函数求导法则，
DL/DWi = DL/DS(S(N(X，W))) * DS/DN(N(X，W)) * DN/DWi(X,W)
计算： 
1、DN/DWi(X,W)=(DN/Dwi1,DN/Dwi2,...,DN/Dwik),而
DN/Dxi1=w11,...,DN/Dxik=w1k, 所以这是W1的转置：DN/DXi(X,W)=np.transpose(W1)，其中W1为W的第1行；
...
同理，DN/Dxik(X,W)=np.transpose(Wk)，其中Wk为W的第k行。
综合上述，DN/DXi(X,W)=np.transpose(W, (1, 0))，这是W的转置。
2、设sigma(N)=F(N),DS/DN(N(X，W))=（DF/DNij）
3、根据上述定义，L(S) 对元素 sij 的导数均为1，故可表示成与S同型的矩阵，元素均为1.
    DL/DS = np.ones_like(S) 。
4、DL/DXi = np.ones_like(S) * DS/DN(N(X，W)) * np.transpose(W, (1, 0))
'''
def grad_03(X:np.ndarray, W:np.ndarray,
           Sigma: Array_Function)-> np.ndarray:
    '计算矩阵函数L=L(S)=L(g(N)),其中N=np.dot(X，W)对于第二个自变量W的导数，其中X的列数必须等于W的行数 '
    assert X.shape[1] == W.shape[0]  #这是矩阵X与W能够相乘的条件，下面的np.dot()是矩阵乘法，结果是矩阵
    N = np.dot(X, W)
    S = Sigma(N)  # 函数值
    # L(S) 将S的所有元素相加：L(S)=s11+s12+s13+...+s32, S=(sij)
    L = np.sum(S)
    #dLdS 根据上述定义，L(S) 对sij 的导数均为1 ！可表示成与S同型的矩阵，元素均为1.
    dLdS = np.ones_like(S)
    # 计算dS/dN 结果是与N同型的矩阵，注意：N的行数与X同，列数与W同。
    dSdN = deriv(Sigma, N)
    # 计算dLdN，结果是矩阵，行数与S同即与N同，也就是与X的行数相同，列数与W相同
    dLdN = dLdS*dSdN
    # dNdX 矩阵，行数=W的列数，列数=W的行数
    dNdX = np.transpose(X, (1, 0))

    # dLdX
    dLdX = np.dot(dNdX, dLdN)
    Z = dLdX
    return Z

input('第二部分结束。下面是第三部分：ok ？')

# II.4 计算的例子
print('-------------例 5-------------')

LW = np.sum(Sigma(np.dot(X, W)))
print('W=:', W, '\n')
print('LW=', LW)
print('------------- 下面是 函数L 关于各 wij 的导数: -------------')
DLW = grad_03(X, W, Sigma)
print(DLW)

## 验算：把w11增加0.01，计算L的增量比，与DL的（1,1）项作比较
W1 = W.copy()
W1[0, 0] += 0.001 # W中把w11增加0.01
delta_LW = np.sum(Sigma(np.dot(X, W1)))-np.sum(Sigma(np.dot(X, W)))

print('------------- 下面是 函数L 关于自变量 w11 的增量比: w11的增量为 0.001 -------------')
# L的增量比
test12 = delta_LW/0.001
print(test12)