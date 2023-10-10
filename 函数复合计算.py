# -*- coding: UTF-8 -*-
###  求复合函数的导数，并用matplotlib作复合函数与导数的图像
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from typing import Callable
from typing import List

#####  第I部分 求实函数及复合函数链并作函数图像

##    I.1  定义5个函数及其产生的函数链
def square(x: np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.power(x, 2)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    ''':relu函数用于ndarray的每个元素'''
    return np.maximum(0.2 * x, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    '''sigmoid函数用于ndarray的每个元素'''
    return 1 / (1 + np.exp(-x))

def sin(x: np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.sin(x)

def cos(x: np.ndarray) -> np.ndarray:
    '''计算ndarray的每个元素的平方'''
    return np.cos(x)

## 定义函数链
Chain1 = [sin, square, leaky_relu, sigmoid, cos]
Chain2 = [sigmoid, square, leaky_relu]
Chain3 = [square, sigmoid]

### I.2 计算函数链的复合函数 (程序中使用了对象Callabel)
# 注意：callable对象是指可以被调用执行的对象，并且可以传入参数。
# 或者说，只要可以在其后面使用小括号来执行代码，那么这个对象就是callable对象。callable对象包括：
# 函数
# 类
# 类里的函数
# 实现了__call__方法的实例对象

# 定义函数组
Array_Function = Callable[[np.ndarray], np.ndarray]
# 定义Chain是函数组成员构成的typing.list列表，称函数链
Chain = List[Array_Function]
print(Chain, '\n', type(Chain))
# typing.List[typing.Callable[[numpy.ndarray], numpy.ndarray]]
# <class 'typing.GenericMeta'>

# 计算2-元链Chain的复合函数
def chain_of_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))

# 计算3-元链Chain的复合函数
def chain_of_3(chain: Chain, a: np.ndarray) -> np.ndarray:
    assert len(chain) == 3
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    return f3(f2(f1(a)))

###  计算l-元素函数链Chain的复合函数
def chain_of_l(chain: Chain, a: np.ndarray) -> np.ndarray:
    lc = len(chain)
    la = len(a)
    FH1 = a
    lst = FH1.reshape(1, la)
    for i in range(lc):
        FH = chain[i](FH1)
        FH1 = FH
        lst = np.append(lst, FH1.reshape(1, la), axis=0)
        print(lst.shape)
    return lst[lc]

#input_range = np.arange(-3, 3, 0.1)
#output_range = chain_of_l(Chain3, input_range)

'''
print(input_range, '\n', len(input_range))
print('--------------------------------------')

print(output_range, '\n' ) #, output_range[1])
'''
####    I.3 作函数链的复合函数图像

## 定义作图方法，，用变量ax来存放AxesSubplot对象
def plot_chain(ax,
               chain: Chain,
               input_range: np.ndarray) -> None:
#Plots a chain function-a function made up of  multiple consecutive ndarray->ndarray mappings-Across the input_range
    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"
    output_range = chain_of_l(chain, input_range)
    ax.plot(input_range, output_range)

## 用Axes对象作图
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))  # 1 Rows, 2 Col，画一个图！
PLOT_RANGE = np.arange(-3, 3, 0.01)

plot_chain(ax[0], Chain3, PLOT_RANGE)
ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title("Function Image for\n$f(x) = sigmoid(square(x))$")

plot_chain(ax[1], Chain2, PLOT_RANGE)
ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title("Function Image for\n$f(x) = sigmoid(square(leaky_relu(x)))$")

plt.show()
