from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.  权重
    - X: A numpy array of shape (N, D) containing a minibatch of data.  小批量训练集
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means 标量
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]    # 类别数
    num_train = X.shape[0]  # 训练数据量
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)    # 得分
        correct_class_score = scores[y[i]]  # 正确类别的得分
        for j in range(num_classes):
            if j == y[i]:   # 是正确的分类，就跳过，这里计算的是svm损失函数
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # my code
                dlds = 1
                dldsy = -1
                # print(X[i].shape)
                # print(X[i].T.shape)
                # print(dW[:,j].shape)
                dW[:, j] += dlds*X[i]  # 非标签类,dW[:,j]意思就是挑出第j列
                dW[:, y[i]] += dldsy*X[i]  # 标签类

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train   # 需要求以下平均值
    dW /= num_train  # 同上，需要求平均值

    # Add regularization to the loss.
    loss += reg*np.sum(W*W)  # 加上正则化损失
    dW += 2*reg*W   # 注意还有一个正则惩罚项要求导
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative（导数）,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)   # 矩阵乘法计算得分
    num_train = X.shape[0]
    num_class = W.shape[1]
    loss = 0.0
    scores -= scores[range(num_train), [y]].T   # 每个分数都减去标签类的得分
    scores += 1.0
    scores[range(num_train), [y]] = 0
    margin = np.maximum(scores, 0)  # 这里的margin已经算是计算之后的max函数了，每一行加起来就是损失分量
    loss = np.sum(margin)/num_train + reg*np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ds = np.zeros_like(margin)
    ds[margin > 0] = 1
    ds[range(num_train), y] -= np.sum(ds, axis=1)
    dW = X.T.dot(ds)
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
