# -*- coding: UTF-8 -*-
'''
    下面是一个简化的线性回归梯度训练模型。
一、模型
    自变量X为m*k矩阵,参数W是k*1矩阵。函数：
     N=X*W， 这里是矩阵乘法 ,结果是 m*1 矩阵；
     P=N+B， 其中B是m*1矩阵，B=(b,...,b)^T,视B为常数
     L=L(P,Y), 其中m*1矩阵Y是目标值，Y=(y1,...,ym)^T, L(P,Y)=MSE(P,Y)是均方差。
     我们要利用 L 对W各元素的偏导数（梯度），“训练”出合适的W，使得误差 L "尽量小"。
     注意，如果视B、Y为常量，那么L可看成如下复合函数的值：
     L=MSE(P,Y),P=P(X,W)=X*W+B,即
     L=L(P(X,W),Y）
二、计算复合函数值的代码                  见：I.1 分步线性回归的前向传递
三、计算梯度的代码                       见：I.2  计算loss对W的梯度：loss_grad()
四、训练的代码，预测结果(训练时使用了 Scikit-Learn自带的数据集 sklearn.datasets 中的
    load_boston ，即“波士顿房价数据集”)。 见: I.3 进行训练的主程序 train() 与训练前准备数据
五、训练举例                            见: I.4
六、对训练模型的评估                      见: I.5
'''
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from typing import Callable
from typing import List, Tuple
from typing import Dict
import pl_chain

##  第I部分 计算复合函数值，其中自变量是2维数组X（对应于一批观测数据），y（对应于目标值）与参数字典（包括 W（权重）和B）。

# I.1 分步线性回归的前向传递
def forward_loss(X_batch: np.ndarray,
              y_batch: np.ndarray,
              weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    分步线性回归的前向传递 (计算复合函数的值)
    输入:m*k的矩阵X表示m行数据，输入m*1的目标向量y，输入参数字典{'W':k*1参数向量，'B':m*1基线向量}
    输出:float数值loss（均方差）和 一个新字典forward_info（保存部分计算结果）
    '''
    # 验证X中的批次和Y的一致
    assert X_batch.shape[0] == y_batch.shape[0]
    # 检查字典weights中的数据：X与W可以相乘
    assert X_batch.shape[1] == weights['W'].shape[0]
    # 检查字典weights中的数据：B是1*1的 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1
    # 计算：
    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(y_batch-P, 2))
    # 保存计算结果
    forward_info: Dict[str, np.ndarry] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return forward_info, loss

# 把1维数组a转换成2维数组输出，代码如下(可以选择输出的数据形如列或形如行)-----程序放在pl_chain.py中
# to_2d_np(a: np.ndarray,  type: str = "col") -> np.ndarray:

# 沿0轴随机排列 X 和 y，二者使用相同的排列 -------程序放在pl_chain.py中
# def permute_data(X: np.ndarray, y: np.ndarray):

#I.2  计算loss对W的梯度：loss_grad()
# loss的定义: loss = np.mean(np.power(y_batch-P(N(X_batch,W),B), 2)) 计算它对参数W的导数，参见第一章笔记中命题2.

# 生成数据函数 generate_batch()
# 输出 X_batch, y_batch
Batch = Tuple[np.ndarray, np.ndarray]
def generate_batch(X: np.ndarray,
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    '''
    Generate batch from X and y, given a start position   注意y是小写字母!
    '''
    assert X.ndim == y.ndim == 2, \
        "X and y must be 2 dimensional 如果y是1维向量，需要先转换成2维的列向量"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]

    return X_batch, y_batch
# 注意：该程序实际上可以用来从一对2维数组（矩阵）X,Y中截取batch_size行数据，从第start+1行开始，到start+batch_size行止。

def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Compute dLdW and dLdB for the step-by-step linear regression model.
    '''
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))

    # need to use matrix multiplication here,
    # with dNdW on the left (see note at the end of last chapter)
    dLdW = np.dot(dNdW, dLdN)

    # need to sum along dimension representing the batch size:
    # see the note near the end of the chapter
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients

# I.3 进行训练的主程序 train() (注意train()需要调用pl_chain.py中的几个程序)

# I.3.1 导入数据集：Boston房价数据集，并进行标准化处理，然后拆分成训练数据集和测试数据集
from sklearn.datasets import load_boston
boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# I.3.2 训练主程序：
def train(X: np.ndarray,
          y: np.ndarray,
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          seed: int = 1) -> None:
    '''
    Train model for a certain number of epochs.
    '''
    # 先确定一个numpy的随机批次，默认的seed=1; 然后给变量start赋初值
    if seed:
        np.random.seed(seed)
    start = 0
    # Initialize weights
    weights = pl_chain.init_weights(X.shape[1])   # 参数W行数等于X的列数，所以它与X的行数无关！
    # Permute data
    X, y = pl_chain.permute_data(X, y)  # 把X,y的行进行一次随机排列

    if return_losses:
        losses = []          # 如果return_losses=T，则准备列表存放每次均方差的值

    for i in range(n_iter):  # 循环1000次

        # Generate batch
        if start >= X.shape[0]:
            X, y = pl_chain.permute_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)  # 从X,y中截取连续的batch_size行
        start += batch_size

        # Train net using generated batch 计算由上述100行数据得到的两个相关量。
        # 即：1）回归模型中变量及复合函数的值构成的字典forward_info（键为：'X','y','N','P'）; 2）损失值loss，是均方差的100倍
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:    # 如果return_weights=T，则函数train()返回上面循环计算的两个结果：列表losses和训练后的weights
        return losses, weights

    return None

# I.4 训练举例 -  The weights (W and B) and the losses, obtained by training 1000 times, by
# custom linear regression model, while the learning_rate=0.001
train_info = train(X_train, y_train,
                   n_iter = 1000,
                   learning_rate = 0.001,
                   batch_size=23,
                   return_losses=True,
                   return_weights=True,
                   seed=180708)
# 训练1000次得到1000个误差（实际上是方差的某个倍数）的1000个数据，即losses；并得到训练后的参数weights
losses = train_info[0]
weights = train_info[1]
# 下面的曲线说明随着训练次数增加，误差的变化情况
plt.title(" The graph of losses trained 1000 times by \ncustom linear regression model")
plt.plot(losses)   # list(range(800)),
plt.show()
# input('OK?')

##  I.5 评估预测情况
print('\n -----------------注意： 本次训练和测试的模型是 线性回归模型------------------------------ ')
# I.5.1 预测函数 predict()
# 把上面训练得到的W和B代入到该函数中，则对于X中的每一行数据，可以计算出其对应预测值，即 N + weights['B']
def predict(X: np.ndarray,
            weights: Dict[str, np.ndarray]):
    '''
    Generate predictions from the step-by-step linear regression model.
    '''

    N = np.dot(X, weights['W'])

    return N + weights['B']
# 对于测试集的数据，计算预测值：
preds = predict(X_test, weights)

# I.5.2 对预测结果的评估：，则对于X中的每一行数据，可以比较其对应的预测值和目标值y_test，并汇总。下面是两种比较算法：
# 一是平均绝对误差mae()函数，另一个是均方根误差rmse()函数
print("Mean absolute error:", round(pl_chain.mae(preds, y_test), 4), "\n"
      "Root mean squared error:", round(pl_chain.rmse(preds, y_test), 4))
# ---------------------------------------------------------------------
# 计算y_test的均值，保留4位小数，然后计算相对于目标值的平均误差率：
pj = np.round(y_test.mean(), 4)
xdpj = np.round(pl_chain.rmse(preds, y_test) / y_test.mean(), 4)
print('\n 目标值即房价的均值为:', pj)
print('\n 相对于目标值的平均误差率为: 百分之', np.round(xdpj*100, 4))
print('-----------------------------------------------------')
# ----------------------------------------------------------
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title(" Predicted vs. Actual values for\ncustom linear regression model")   # ;
plt.xlim([0, 51])
plt.ylim([0, 51])
plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51])
plt.show()
# plt.savefig(GRAPHS_IMG_FILEPATH + "01_linear_custom_pred_vs_actual.png")

# I.5.3 分析最重要的特征
#--------------------------------------
w = np.round(weights['W'].reshape(-1), 4)
print('训练1000次得到的参数W是：\n W=', w, '\n'
      '其绝对值最大的元素是W[12]')
# 因为W[12]的绝对值最大，所以推断，在该数据集中，其相应的列-即第13列，或者说第13个特征-应该是一个最重要的特征。
# 下面的散点图描述了：当改变测试集的第13列，让它的值在[-1.5,3.5]区间增加时，相应的目标值的大小出现了什么变化。
NUM = 40         # 分40步进行线性插值，
# 下面repeat的语法是：numpy.repeat(a-被重复的内容, repeats-重复次数, axis=None-沿哪个维度重复)，
a = np.repeat(X_test[:, :-1].mean(axis=0, keepdims=True), NUM, axis=0)  # 在测试集中取出40行处理过的数据，每行都只取前12列（注意：
# 这里把每一个数据都用其所在列的均值代替了，所以a的40行数据是重复的）。
b = np.linspace(-1.5, 3.5, NUM).reshape(NUM, 1) # 把（-1.5，3.5）等分出40个点，得到一列数据(40,1)
#print('a=:\n ', a, '\n', a.shape, '\n', a[25, :])
#input('OK?')
test_feature = np.concatenate([a, b], axis=1)  # 按列把数组a与b进行拼接，拼接后第13列为b构成新数据集，
test_preds = predict(test_feature, weights)[:, 0]  # 用新数据集和训练得到的权重计算预测值
#print('数据集最后一列的型：\n', np.shape(X_test[:, 12]))
plt.scatter(X_test[:, 12], y_test)  # 作散点图：以数据集的最后一列元素值（不是b！）与相应行的目标值为坐标确定152个点
plt.plot(np.array(test_feature[:, -1]), test_preds, linewidth=2, c='orange') # 以b为横坐标值，相应的新预测值为纵坐标值作图
plt.ylim([6, 51])
plt.xlabel("Most important feature (normalized)")
plt.ylabel("Target/Predictions")
plt.title("Most important feature vs. target and predictions,\n custom linear regression");
# plt.savefig(GRAPHS_IMG_FILEPATH + "03_most_important_feature_vs_predictions.png")
plt.show()

