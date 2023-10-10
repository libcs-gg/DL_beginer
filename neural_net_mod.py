# -*- coding: UTF-8 -*-
'''
    下面是一个简单的神经网络训练模型，用 Boston房价数据来分析预测 Boston房价。包括六个部分
一、准备 Boston 数据集
    使用 Scikit-Learn自带的数据集 sklearn.datasets 中的 load_boston ，即“波士顿房价数据集”
二、建立模型
输入变量： X为m*k矩阵,参数W1是k*n矩阵。通过矩阵乘积 M1=X*W1，对每个观测值（X的一行，原始特征），都得出 n 个
    加权和（相当于n个线性回归）；
    偏差矩阵为B1,shape(B1)=(1,n),即矩阵 X*W1 的每一列对应着B1的一个数，二者相加是广播运算；
    类似地，参数W2是n*1矩阵而B2的型为(1,1)，目标值y是m*1矩阵，X的每一行，对应y的一个元素。
函数：N1 = M1+B1,注意M1=X*W1是矩阵积,故N1是 m*n 矩阵；O1 = sigma(N1), 也是 m*n 矩阵；
     P = M2+B2，其中 M2=O1*W2，是矩阵乘法，相当于原始特征的加权和通过非线性函数sigma作用后再进行一次线性回归；
     L=L(P,y), 其中m*1矩阵y是目标值，y=(y1,...,ym)^T, L(P,y)=MSE(P,y)是均方差。
目标：利用 L 对W各元素的偏导数（梯度），“训练”出合适的W，使得误差 L小于预期值(越小越好)。
     注意，如果视B1,B2,Y为常量，那么L可看成如下复合函数的值：
     L=MSE(P,Y),其中 P=P(X,W1,W2)=O*W2+B2=[sigma(X*W1+B1)]*W2+B2,即 L=L(P(X,W),Y）
三、计算复合函数值的代码
四、计算梯度的代码
五、训练的代码，预测结果
六、评估
'''
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from typing import List, Tuple
from typing import Dict
import pl_chain
from sklearn.metrics import r2_score

np.set_printoptions(precision=4) # 设置输出4位小数

## I.1 导入Boston数据集，并进行预处理

# I.1.1 导入数据集：
from sklearn.datasets import load_boston
boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

# I.1.2 对数据进行标准化：
'''
StandardScaler()是一个标准化数据的方法，保证每个维度数据方差为1.均值为0，以使预测结果不会被某些维度过大的特征值主导。
公式：$$ x^* = \frac{x - \mu}{\sigma} $$ 
很显然，它只是进行转换，只是把训练数据转换成标准的正态分布
'''
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)
'''
数据预处理中的方法：
1、Fit(): Method calculates the parameters μ and σ and saves them as internal objects.
求data的均值，方差，最大最小值等统计量。
2、Transform(): Method using these calculated parameters apply the transformation to a particular dataset.
在Fit的基础上应用均值、方差进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。
3、Fit_transform(): joins the fit() and transform() method for transformation of dataset(是fit和transform的组合)
StandardScaler().fit_transform(data)对data进行标准化,使得其每一列（对应一个特征）都~N(0,1)，即服从标准正态分布。
作用：
transform()和fit_transform()的功能都是对数据进行某种统一处理，例如进行标准化~N(0,1)，或将数据缩放(映射)到某个固定区间归一化，或正则化。
注意事项：
1 训练集数据和测试集数据要使用同一个方法处理；
2 使用fit_transform()时，必须先处理训练集数据后处理测试集数据.
'''

# I.1.3 数据集拆分出训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# I.1.4 生成数据函数 generate_batch()
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
        "X and y must be 2 dimensional 如果y是1维向量，需要先转换成2维的数组，是列向量"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]

    return X_batch, y_batch
# 该程序用来从一对2维数组X和Y中截取batch_size行数据，从第start行开始，到start+batch_size-1行止。


## I.2 计算分步神经网络模型的前向传递结果 (计算复合函数的值)forward_info和损失值loss
def forward_loss(X_batch: np.ndarray,
              y_batch: np.ndarray,
              weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    计算分步神经网络模型的前向传递结果 (计算复合函数的值)forward_info和损失值loss
    输入:m*k的矩阵X表示m行数据，输入m*1的目标向量y，输入参数字典{'W1':k*n参数向量，'W2':n*1向量,'B1':1*1矩阵, 'B2':1*1矩阵}
    输出:float数值loss（均方差）和 一个新字典forward_info（保存部分计算结果）
    '''
    # 验证X中的批次和Y的一致
    assert X_batch.shape[0] == y_batch.shape[0]
    # 检查字典weights中的数据：X与W可以相乘
    assert X_batch.shape[1] == weights['W1'].shape[0]
    # 检查字典weights中的数据：B是1*1的 ndarray
    assert weights['B2'].shape[0] == weights['B2'].shape[1] == 1
    # 计算：
    M1 = np.dot(X_batch, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = pl_chain.sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']
    loss = np.mean(np.power(y_batch-P, 2))
    # 保存计算结果
    forward_info: Dict[str, np.ndarry] = {}
    forward_info['X'] = X_batch
    forward_info['N1'] = N1
    forward_info['M1'] = M1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return forward_info, loss

##I.3  计算loss 对W1 W2 B1 B2 的梯度：loss_grad()
# loss的定义: loss = np.mean(np.power(y_batch-P), 2)) 计算它对参数 W1 等的导数，参见第一章笔记中命题2.
# Note: All the parameters W1 W2 B1 B2 be included in the dictionary weights .
# 为检查计算方便，下列矩阵均为2维数组，其shape如下：
# shape(X)=(m,k),shape(W1)=(k,n),shape(M1)=(m,n),shape(B1)=(1,n)-对应于W1的每列设定一个基数，当n=1时B1为一个数，
# shape(N1)=(m,n),shape(O1)=(m,n),
# shape(W2)=(n,1), shape(M2)=(m,1),shape(B2)=(1,1),shape(P)=(m,1),shape(y)=(m,1),shape(L)=(1,1) 。
def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Compute 梯度 dLdW1,dLdW2 and dLdB1,dLdB2 for the neural network model.
    '''
    batch_size = forward_info['X'].shape[0]

    dLdP = -1 * (forward_info['y'] - forward_info['P'])   # P和y的型都是（m，1），此项型为(m,1) 。
    dLdy = np.ones_like(forward_info['y'])                # 此项型也是 (m,1) 。

    dPdM2 = np.ones_like(forward_info['M2'])         # P(M2,B2)是数组M2和B2的成员求和，其所有偏导数均为1,其型同shape(M2)=(m,1)
    dPdB2 = np.ones_like(weights['B2'])              # shape=shape(B2)
    dLdM2 = dLdP*dPdM2                               #  它们的型都是（m，1）

    dM2dO1 = np.transpose(weights['W2'], (1, 0))     # 注意2维数组 W2 型为 （n,1），所以其转置的型为（1,n）。
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))  # 注意2维数组O1的型为（m,n），所以其转置的型为：（n,m）。

    # 下面是sigmoid()函数的导数,注意shape(N1)=(m,n)，所以 dO1dN1 的型也是(m,n)。
    dO1dN1 = pl_chain.sigmoid(forward_info['N1'])*(1-pl_chain.sigmoid(forward_info['N1']))

    dN1dB1 = np.ones_like(weights['B1'])            # 注意2维数组 B1 型为 （1,n）。
    dN1dM1 = np.ones_like(forward_info['M1'])       # 注意2维数组 M1 型为 （m,n）。

    dO1dM1 = dO1dN1*dN1dM1                          # 注意两个2维数组的型均为（m,n），故对应元素之积后型不变。

    dM1dW1 = np.transpose(forward_info['X'], (1, 0))     # 注意2维数组 X 型为 （m,k）,所以其转置的型为（k,m）。

    # dO1dW1 = np.dot(dM1dW1, dO1dM1)                 # 注意(k,m)矩阵与（m,n）矩阵的积，结果是(k,n）矩阵。
    dLdO1 = dLdM2 * dM2dO1             #dLdM2型为(m,1)，与型为(1,n)的dM2dO1做广播相乘后得(m,n)矩阵，此处也可以是矩阵积
    dLdM1 = dLdO1*dO1dM1               # 二者的型均为(m,n)。
    # need to use matrix multiplication here,
    # with dM1dW1 on the left (see note at the end of last chapter)
    dLdW1 = np.dot(dM1dW1, dLdM1)            # 注意(k,m)矩阵与（m,n）矩阵的矩阵积，结果是(k,n）矩阵。
    dLdB1 = ((dLdO1*dO1dN1)*dN1dB1)          # 前者型为 (m,n)，后者型为(1,n)，广播相乘后型为(m,n)。
    # need to sum along dimension representing the batch size:
    # see note near the end of the chapter
    dLdB2 = (dLdP * dPdB2)         # 前者型为 (m,1)，后者型为(1,1)，广播相乘后型为(m,1)。
    dLdW2 = np.dot(dM2dW2, dLdM2)   # 矩阵dM2dW2的型为(n,m),而矩阵dLdM2的型为(m,1),矩阵积的型为 (n,1) !


    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W1'] = dLdW1
    loss_gradients['W2'] = dLdW2
    loss_gradients['B1'] = dLdB1.sum(axis=0)
    loss_gradients['B2'] = dLdB2.sum(axis=0)

    return loss_gradients

##I.4 初始化参数函数Init_weights()  和 预测函数predict()
def init_weights(input_size: int,
                 hidden_size: int) -> Dict[str, np.ndarray]:
    '''
    Initialize weights during the forward pass for step-by-step neural network model.
    '''
    weights: Dict[str, np.ndarray] = {}
    weights['W1'] = np.random.randn(input_size, hidden_size)    # input_size为模型中的k,即特征数,hidden_size为模型中的n
    weights['B1'] = np.random.randn(1, hidden_size)             # 型为(1,n)
    weights['W2'] = np.random.randn(hidden_size, 1)             # 型为(n,1)
    weights['B2'] = np.random.randn(1, 1)
    return weights

def predict(X: np.ndarray,
            weights: Dict[str, np.ndarray]) -> np.ndarray:
    '''
    Generate predictions from the step-by-step neural network model.
    '''
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = pl_chain.sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']

    return P

##I.5 训练
#I.5.1 训练函数
def train(X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray,
          n_iter: int = 1000,
          test_every: int = 1000,
          learning_rate: float = 0.01,
          hidden_size=13,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          return_scores: bool = False,
          seed: int = 1) -> None:
    if seed:
        np.random.seed(seed)

    start = 0

    # Initialize weights
    weights = init_weights(X_train.shape[1],
                           hidden_size=hidden_size)

    # Permute data
    X_train, y_train = pl_chain.permute_data(X_train, y_train)

    losses = []

    val_scores = []      # 用来记录

    for i in range(n_iter):

        # Generate batch
        if start >= X_train.shape[0]:
            X_train, y_train = pl_chain.permute_data(X_train, y_train)
            start = 0

        X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
        start += batch_size

        # Train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

        if return_scores:
            if i % test_every == 0 and i != 0:
                preds = predict(X_test, weights)
                val_scores.append(r2_score(preds, y_test))

    if return_weights:
        return losses, weights, val_scores

    return None

# I.5.2 训练举例 训练10000次并画出误差值的图像:

train_info = train(X_train, y_train, X_test, y_test,
                   n_iter=10000,
                   test_every = 1000,
                   learning_rate = 0.001,
                   batch_size=23,
                   return_losses=True,
                   return_weights=True,
                   return_scores=False,
                   seed=180807)
losses = train_info[0]
weights = train_info[1]

plt.figure(figsize=(30, 30), dpi=80)
plt.title('" The graph of losses trained 10000 times by \n neural network model"')  # 图片标题
plt.xlabel('trained times')  # x轴变量名称
plt.ylabel('losses')  # y轴变量名称
plt.plot(list(range(10000)), losses, label='the curve of training effect ')  # 画出 a_line 线  label="x": 图中左上角示例
plt.legend()  # 画出曲线图标
#plt.savefig('1.jpg') # 图片保存
plt.show()  # 画出图像

# I.6 评估与考察
# I.6.1 Investigation of most important features 考察最重要的特征
np.set_printoptions(precision= 3)     #设置输出的小数位数=3
predicts = predict(X_test, weights)
# Most important combinations of features are the two with absolute values of greater than 9:
ww = weights['W2']
print('\n -----------------注意： 本次训练和测试的模型是 分步神经网络模型------------------------------ ')
print('\n 下面考察训练得到的参数W2和W1：\n W2是第二组参数，型为(n,1), W2.T=:\n', ww.T,
      '\n 其中绝对值最大的两个分别是：第8个和第10个，此二者应该为 最重要的特征。'
      '\n 参数W1的型为(k,n),其中对应W1中绝对值最大的两个参数的列分别是：第8列和第10列。')
w = weights['W1']
w_important1 = w[7]
w_important2 = w[9]
print('\n W1的第8列为：\n', w_important1, '\n W1的第10列为：\n', w_important2)

# I.6.2 预测的误差分析
print('\n 下面是误差情况：\n')
print("Mean absolute error-平均绝对误差:", round(pl_chain.mae(predicts, y_test), 4), "\n"
      "Root mean squared error-均方根误差:", round(pl_chain.rmse(predicts, y_test), 4),
      '\n 而均方差是：', np.round(np.mean(np.array(np.power(predicts - y_test, 2))), 4))
plt.xlabel("Predicted value")
plt.ylabel("Target")
plt.title("Predicted value vs. target,\n neural network regression")
plt.xlim([0, 51])
plt.ylim([0, 51])
plt.scatter(predicts, y_test)    # 画散点图，以(predicts, y_test)为坐标
plt.plot([0, 51], [0, 51])       # 画直线图，直线为 x=y
plt.show()
# plt.savefig(GRAPHS_IMG_FILEPATH + "07_neural_network_regression_preds_vs_target.png");
# print('\n', np.round(np.mean(np.array(np.abs(predicts - y_test))), 4))
# print('\n 均方差是：', np.round(np.mean(np.array(np.power(predicts - y_test, 2))), 4))

# I.6.3  最重要的特征分析
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
plt.scatter(X_test[:, 12], predicts)  # 作散点图：以数据集的最后一列元素值（不是b！）与相应行的预测值为坐标确定152个点
plt.plot(np.array(test_feature[:, -1]), test_preds, linewidth=2, c='orange') # 以b为横坐标值，相应的新预测值为纵坐标值作图
plt.ylim([6, 51])
plt.xlabel("Most important feature (normalized)")
plt.ylabel("Target/Predictions")
plt.title("Most important feature vs. target and predictions,\n neural network regression");
# plt.savefig(GRAPHS_IMG_FILEPATH + "03_most_important_feature_vs_predictions.png")
plt.show()