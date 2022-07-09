from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    dW = np.zeros_like(W)
    # scores=np.zeros(num_train,num_class) 得分不需要保存
    for i in range(num_train):
        scores = X[i].dot(W)  # 得到了一个长度为 10 的向量
        scores -= np.max(scores)  # 进行一次平移操作，以免发生数值爆炸
        scores = np.exp(scores)   # 用e进行归一化
        loss+=-np.log(scores[y[i]]/np.sum(scores))/num_train # 计算损失函数
        dW[:, y[i]] -= X[i]
        for j in range(num_class):
            dW[:, j] += scores[j]*X[i]/np.sum(scores)  # 非标签类,dW[:,j]意思就是挑出第j列
    # 不知道为什么，上面这种形式的误差非常大
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # num_train=X.shape[0]
    # num_class=W.shape[1]
    # scores = X.dot(W) # 得到分数
    # scores-=np.max(scores,axis=1,keepdims=True) # 进行平移
    # f=np.exp(scores)  # 利用e进行归一化
    # loss=np.sum(-np.log(f[range(num_train),y]/np.sum(f,axis=1)))/num_train+reg*np.sum(W**2) #计算损失
    # for i in range(num_train):
    #   dW[:,y[i]]-=X[i]
    #   for j in range(num_class):
    #     dW[:,j]+=X[i]*f[i][j]/np.sum(f[i])
    loss+=reg*np.sum(W**2)
    dW/=num_train
    dW+=2*W*reg
    # 虽然不知道什么情况，但是确实弄好了iai
    # 一步一步来，不要着急，不要忘记正则化也要算上去
    # 先进行求平均，再加上正则化
    return loss, dW




def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    num_class=W.shape[1]
    scores=X.dot(W)
    scores-=np.max(scores,axis=1,keepdims=True)
    f=np.exp(scores)  # f 的内容是e^s
    loss=np.sum(-np.log(f[range(num_train),y]/np.sum(f,axis=1)))/num_train+reg*np.sum(W**2)
    ds=f/np.sum(f,axis=1,keepdims=True)
    ds[range(num_train),y]-=1
    dW=X.T.dot(ds)
    dW/=num_train
    dW+=2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
