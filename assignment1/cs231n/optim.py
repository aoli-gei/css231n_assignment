import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
该文件实现了用于训练神经网络的各种常用一阶更新规则。每个更新规则都接受当前权重和
损失相对于这些权重的梯度并产生下一组权重。每个更新规则都有相同的接口:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.
对于大多数更新规则，默认学习率可能不会
表现良好;但是其他超参数的默认值应该
适用于各种不同的问题。

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
为了提高效率，更新规则可以执行就地更新，改变 w 和
设置 next_w 等于 w。
"""

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    执行普通随机梯度下降

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    ###########################################################################
    # TODO: Implement the vanilla stochastic gradient descent update formula. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    w-=dw*config['learning_rate']

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return w, config