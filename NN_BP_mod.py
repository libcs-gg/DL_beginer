# -*- coding: UTF-8 -*-
'''
本文件是一个基于BP算法的神经网络训练模型，使用Operation类、Layer类、NeuralNetwork类、Loss类、Optimizer类和Train类来实现（参见Weidman第3章）。
(==注：算法简介-BP（Back Propagation）网络是1986年由Rumelhart和McCelland为首的科学家小组提出，是一种按误差逆传播算法训练的多层
前馈网络，也是目前应用最广泛的神经网络模型之一。
BP网络能学习和存贮大量的输入-输出模式映射关系，而无需事前揭示描述这种映射关系的数学方程。它的学习规则是使用最速下降法，通过反向传播来不断调整网络的
权值和阈值，使网络的误差平方和最小。
BP神经网络模型拓扑结构包括输入层（input）、隐藏层(hide layer)和输出层(output layer)。在模拟过程中收集系统所产生的误差，通过误差反传，然后
调整权值大小，通过该不断迭代更新使得模型趋于整体最优化。
--
BP神经网络具有以下优点：
1 非线性映射能力：BP实质上实现了从输入到输出的映射功能，数学理论证明三层的神经网络就能够以任意精度逼近任何非线性连续函数。这使得其特别适合
于求解内部机制复杂的问题，即BP神经网络具有较强的非线性映射能力。
2 自学习和自适应能力：BP神经网络在训练时，能够通过学习自动提取输入、输出数据间的“合理规则”，并自适应地将学习内容记忆于网络的权值中。
3 泛化能力：所谓泛化能力是指在设计模式分类器时，既要考虑网络在保证对所需分类对象进行正确分类，还要关心网络在经过训练后，能否对未见过的模式或有噪声污染
的模式，进行正确的分类。也即BP神经网络具有将学习成果应用于新知识的能力。
--
BP神经网络具有以下缺点点：
1 局部极小化问题：从数学角度看BP神经网络为局部搜索的优化方法。网络的权值沿局部改善的方向逐渐调整，这样会使算法陷入局部极值导致网络训练失败。
加上BP神经网络对初始网络权重非常敏感，以不同的权重初始化网络，其往往会收敛于不同的局部极小，这也是每次训练得到不同结果的根本原因。
2 收敛速度慢：由于BP神经网络算法本质上为梯度下降法，它所要优化的目标函数是非常复杂的，因此，必然会出现“锯齿形现象”，这使得BP算法低效；又由于优化的
目标函数很复杂，它必然会在神经元输出接近0或1的情况下，出现一些平坦区，在这些区域内，权值误差改变很小，使训练过程几乎停顿；BP神经网络模型中，
为了使网络执行BP算法，不能使用传统的一维搜索法求每次迭代的步长，而必须把步长的更新规则预先赋予网络，这种方法也会引起算法低效。以上种种，
导致了BP神经网络算法收敛速度慢的现象。
3 BP 神经网络结构选择不一：BP神经网络结构的选择至今尚无一种统一而完整的理论指导，一般只能由经验选定。网络结构选择过大，训练中效率不高，
可能出现过拟合现象，造成网络性能低，容错性下降，若选择过小，则又会造成网络可能不收敛。而网络的结构直接影响网络的逼近能力及推广性质。
因此，应用中如何选择合适的网络结构是一个重要的问题。
==）
本文内容分以下十个部分：
I. 神经网络的 Operation 和 ParamOperation 类
II. 具体运算：三种运算子类（矩阵乘法、加法、Sigmoid()函数的作用）
III. 层和稠密层 Layer and Dense
IIII。Loss类及其子类 MeanSquaredError
V。神经网络对应的类 NeuralNetwork(object) （三个神经网络的例子）
VI.优化类和SGD类
VII.训练类
VIII.误差计算和评估函数
VIIII.数据准备-波士顿Boston房产数据集：使用 Scikit-Learn自带的数据集 sklearn.datasets 中的 load_boston ，即“波士顿房价数据集”
X.三个训练的例子
模型与记号：
输入变量： X为m*k矩阵,参数W1是k*n矩阵。通过矩阵乘积 M1=X*W1，对每个观测值（X的一行，原始特征），都得出 n 个 加权和，
偏差矩阵为B1,shape(B1)=(1,n),即矩阵 X*W1 的每一列对应着B1的一个数，二者相加是广播运算；类似地，参数W2是n*1矩阵而B2的型为(1,1)，
目标值y是m*1矩阵，X的每一行，对应y的一个元素。
函数：N1 = M1+B1,注意M1=X*W1是矩阵积,故N1是 m*n 矩阵；O1 = sigma(N1), 也是 m*n 矩阵；
     P = M2+B2，其中 M2=O1*W2，是矩阵乘法，相当于原始特征的加权和通过非线性函数sigma作用后再进行一次线性回归；
     L=L(P,y), 其中m*1矩阵y是目标值，y=(y1,...,ym)^T, L(P,y)=MSE(P,y)是均方差。
目标：利用 L 对 W各元素的偏导数（梯度），“训练”出合适的W，使得误差 L很小。注意，如果视B1,B2,Y为常量，那么L可看成如下复合函数的值：
     L=MSE(P,Y),其中 P=P(X,W1,W2)=O*W2+B2=[sigma(X*W1+B1)]*W2+B2,即 L=L(P(X,W),Y）
'''
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Callable
from typing import List, Tuple, Dict
import pl_chain
from sklearn.metrics import r2_score

# 下面函数用来检测两个输入的数组a,b是否同型。若不同型则报错，并提示两个数组的型。
def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape; 
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array.shape), tuple(array_grad.shape))
    return None

## I. 神经网络的 Operation 和 ParamOperation 类

# I.1 神经网络的运算类
class Operation(object):
    '''
    Base class for an "operation" in a neural network.
    '''
    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        The _input_grad method must be defined for each Operation
        '''
        raise NotImplementedError()

# I.2 运算类的子类--带参数的运算类
class ParamOperation(Operation):
    '''
    An Operation with parameters.
    '''

    def __init__(self, param: ndarray) -> ndarray:
        '''
        The ParamOperation method
        '''
        super().__init__()
        self.param = param

# 下面函数由输出的梯度计算输入的梯度
    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        '''

        assert_same_shape(self.output, output_grad)   # 输出与梯度输出必须同型

        self.input_grad = self._input_grad(output_grad)  # 由输出梯度计算推导得到输入梯度
        self.param_grad = self._param_grad(output_grad)  # 由输出梯度计算推导得到参数梯度

        assert_same_shape(self.input_, self.input_grad)  # 输入梯度与输入必须同型
        assert_same_shape(self.param, self.param_grad)   # 参数梯度与参数必须同型

        return self.input_grad          # 返回输入梯度

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Every subclass of ParamOperation must implement _param_grad.
        '''
        raise NotImplementedError()


## II. 特殊的类 Specific Operations

# II.1 参数类的子类--权重计算类
class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network.
    '''

    def __init__(self, W: ndarray):
        '''
        Initialize Operation with self.param = W.
        '''
        super().__init__(W)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray)  -> ndarray:
        '''
        Compute parameter gradient.
        '''
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


# II.2 参数类的子类-- 偏差项Bias的加入
class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''

    def __init__(self,
                 B: ndarray):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

# II.3 运算类的子类-- Sigmoid激活函数类和线性
class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Pass through'''
        return output_grad

## III. 层和稠密层 Layer and Dense

# III.1 层的类
class Layer(object):
    '''
    A "layer" of neurons in a neural network.(神经网络中的一个神经元的‘层’)
    '''

    def __init__(self,
                 neurons: int):
        '''
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        “神经元”的数量大致对应于层的“宽度”
        '''
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:  #注： 此处的 int 应该是 ndarray，我修改后运行通过。#  ndarray) -> None:
        '''
        The _setup_layer function must be implemented for each layer（每层必须执行此函数，它接收一个输入-整数）
        '''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        '''
        Passes input forward through a series of operations
        通过一系列运算类把输入向前传递
        '''
        if self.first:
            self._setup_layer(input_)              ##  按函数 _setup_layer() 的定义它接收int ??
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Passes output_grad backward through a series of operations
        Checks appropriate shapes
        '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        '''
        Extracts the _param_grads from a layer's operations
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        '''
        Extracts the _params from a layer's operations
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

# III.2 层的子类--稠密层的类

class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None

## IIII. Loss and MeanSquaredError 类

# IIII.1  Loss类
class Loss(object):
    '''
    The "loss" of a neural network
    '''

    def __init__(self):
        '''Pass'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Computes the actual loss value
        '''
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
        Computes gradient of the loss value with respect to the input to the loss function
        '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
        Every subclass of "Loss" must implement the _output function.
        '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
        Every subclass of "Loss" must implement the _input_grad function.
        '''
        raise NotImplementedError()

# IIII.2  MeanSquaredError 类
class MeanSquaredError(Loss):

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
        Computes the per-observation squared error loss
        '''
        loss = (
            np.sum(np.power(self.prediction - self.target, 2)) /
            self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:
        '''
        Computes the loss gradient with respect to the input for MSE loss
        '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]

## V. 神经网络类
class NeuralNetwork(object):
    '''
    The class for a neural network.
    '''
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss,
                 seed: int = 1) -> None:
        '''
        Neural networks need layers, and a loss.
        '''
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)
# 注：setattr() 函数对应于函数 getattr()，用于设置对象的属性值，该属性不一定是对象存在的，可以给对象设置新的属性并赋值。
    # setattr() 语法：setattr(object, name, value)      #

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
        Passes data forward through a series of layers.
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
        Passes data backward through a series of layers.
        '''

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:
        '''
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        '''

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
        Gets the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Gets the gradient of the loss with respect to the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.param_grads

## VI. 优化类和SGD类 Optimizer and SGD
class Optimizer(object):
    '''
    Base class for a neural network optimizer.
    '''
    def __init__(self,
                 lr: float = 0.01):
        '''
        Every optimizer must have an initial learning rate.
        '''
        self.lr = lr

    def step(self) -> None:
        '''
        Every optimizer must implement the "step" function.
        '''
        pass

class SGD(Optimizer):
    '''
    Stochasitc gradient descent optimizer.
    '''
    def __init__(self,
                 lr: float = 0.01) -> None:
        '''Pass'''
        super().__init__(lr)

    def step(self):
        '''
        For each parameter, adjust in the appropriate direction, with the magnitude of the adjustment
        based on the learning rate.
        '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param -= self.lr * param_grad

## VII. 训练类

from copy import deepcopy
from typing import Tuple

class Trainer(object):
    '''
    Trains a neural network
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        '''
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        '''
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> Tuple[ndarray]:
        '''
        Generates batches for training
        '''
        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch


    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True) -> None:
        '''
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = pl_chain.permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}""")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break

## VIII. 误差计算和评估函数 Evaluation metrics; 三个神经网络模型：lr, nn, dl

def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("平均绝对误差Mean absolute error为: {:.2f}".format(mae(preds, y_test)))
    print()
    print("均方根误差Root mean squared error为 {:.2f}".format(rmse(preds, y_test)))

lr = NeuralNetwork(
    layers=[Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

## VIIII. 数据准备--波士顿房产数据集

from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

# Scaling the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

def to_2d_np(a: ndarray,
          type: str="col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)


## X. 三个训练的例子

# helper function
def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

# X.1 第一个例子
print('--------训练：第一个例子 lr网络--------------')
trainer = Trainer(lr, SGD(lr=0.01))
# 定义训练类实例 "trainer"，使用神经网络 lr(线性回归网络) , 优化器SGD，学习率lr=0.01；
# 下面调用方法fit并输入 数据，然后周期地打印loss的结果；
# 最后评估训练效果，用预测数据与测试数据比较所得的 “平均绝对误差” 和 “均方根误差”
trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501)
print()
eval_regression_model(lr, X_test, y_test)

print('---------训练：第二个例子 nn网络-------------')

# X.2 第二个例子
trainer = Trainer(nn, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 100,
       eval_every = 10,
       seed=20190501);
print()
eval_regression_model(nn, X_test, y_test)

print('-----------训练：第三个例子 dl网络-------------')
# X.3 第三个例子
trainer = Trainer(dl, SGD(lr=0.01))
# 定义训练类实例 "trainer"，使用神经网络 dl(深度学习网络) , 优化器SGD，学习率lr=0.01；
# 下面调用方法fit并输入 数据，然后周期地打印loss的结果；
# 最后评估训练效果，用预测数据与测试数据比较所得的 “平均绝对误差” 和 “均方根误差”
trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 100,
       eval_every = 10,
       seed=20190501)
print()
eval_regression_model(dl, X_test, y_test)