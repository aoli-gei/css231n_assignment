from builtins import range
from matplotlib.pyplot import axis
from pyparsing import stringEnd
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_vector = x.reshape(x.shape[0], -1)
    out = x_vector.dot(w)
    out += b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T).reshape(x.shape)
    x_vector = x.reshape(x.shape[0], -1)
    # print(dx.shape)
    dw = x_vector.T.dot(dout).reshape(w.shape)
    db = np.sum(dout, axis=0)  # 注意画图

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.reshape(x, (x.shape[0], -1))  # 先把他拆分成单个向量
    out = np.maximum(0, x)
    out.reshape(x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mask = np.int64(x > 0)
    dx = dout * mask

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    scores = x - np.max(x, axis=1, keepdims=True)  # 进行平移
    f = np.exp(scores)  # 用e进行归一化
    normalized_f = f / np.sum(f, axis=1, keepdims=True)
    loss = np.sum(-np.log(f[range(num_train), y] /
                          np.sum(f, axis=1))) / num_train

    normalized_f[range(num_train), y] -= 1
    dx = normalized_f / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential指数 decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    论文中使用的样本均值和方差是用一个大训练数据集的均值和方差，而在本实验我们用的是正在运行的均值和方差

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        std = np.sqrt(sample_var + eps)
        x_norm = (x - sample_mean) / std
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, x_norm, gamma, sample_mean, sample_var, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        std = np.sqrt(running_var + eps)
        x_norm = (x - running_mean) / std
        out = gamma * x_norm + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (x,x_norm,gamma,sample_mean,sample_var,eps)
    x, x_norm, gamma, sample_mean, sample_var, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)  # 注意这个 *gamma 不是点乘
    dbeta = np.sum(dout, axis=0)  # 求出 dbeta
    dLdx_norm = dout * gamma
    dLdvar = np.sum(dLdx_norm * (-0.5 * (x - sample_mean) *
                                 ((sample_var + eps)**(-1.5))),
                    axis=0)
    dLdu = np.sum(-1 / np.sqrt(sample_var + eps) * dLdx_norm, axis=0)
    dLdx = dLdx_norm*(1/np.sqrt(sample_var+eps))+dLdvar * \
        (2/x.shape[0])*(x-sample_mean)+dLdu/x.shape[0]

    dx = dLdx

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, gamma, sample_mean, sample_var, eps = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)

    std = np.sqrt(sample_var + eps)
    dLdx_norm = dout * gamma

    dLdvar = np.sum(dLdx_norm * -0.5 * (x - sample_mean) * (std**-3), axis=0)
    dLdu = np.sum(dLdx_norm * -1 / std, axis=0)

    dx = dLdx_norm / std + dLdvar * 2 / N * (x - sample_mean) + dLdu / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_mean = np.mean(x, axis=1, keepdims=True)
    x_var = np.var(x, axis=1, keepdims=True)
    std = np.sqrt(x_var + eps)
    x_norm = (x - x_mean) / std
    out = gamma * x_norm + beta

    cache = (x, x_norm, gamma, x_mean, x_var, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (x,x_norm,gamma,x_mean,x_var,eps)
    x, x_norm, gamma, x_mean, x_var, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)  # 注意这个 *gamma 不是点乘
    dbeta = np.sum(dout, axis=0)  # 求出 dbeta

    D = 1.0 * x.shape[1]
    dldx_norm = dout * gamma
    dldvar = np.sum(-0.5 * dldx_norm * (x - x_mean) * ((x_var + eps)**-1.5),
                    axis=1)
    dldu = np.sum(-dldx_norm / np.sqrt(x_var + eps), axis=1)
    dx = dldx_norm/np.sqrt(x_var+eps)+dldu.reshape(-1, 1) / \
        D+dldvar.reshape(-1, 1)*2/D*(x-x_mean)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        p = dropout_param['p']
        mask = np.random.rand(*x.shape) < p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = x
    N = x.shape[0]  # 图片数量
    F = w.shape[0]  # 滤波器数量
    C = w.shape[1]  # 通道数量

    H = x.shape[2]  # 输入 x 的高度
    W = x.shape[3]  # 输入 x 的宽度

    HH = w.shape[2]  # 滤波器的高度
    WW = w.shape[3]  # 滤波器的宽度

    # 是否需要填充？
    if pad != 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                       'constant')  # 对每一个通道进行填充

    # # 检查填充效果
    # print(x.shape)
    # print(x_pad.shape)

    H_pad = x_pad.shape[2]
    W_pad = x_pad.shape[3]

    # 若无法整除，直接返回错误
    assert (H + 2 * pad - HH) % stride == 0
    assert (W + 2 * pad - WW) % stride == 0

    H_out = 1 + (H + 2 * pad - HH) // stride  # 输出特征图的高度
    W_out = 1 + (W + 2 * pad - WW) // stride  # 输出特征图的宽度

    out = np.zeros((N, F, H_out, W_out))  # 初始化特征图

    loc_x = 0
    loc_y = 0
    for index in range(N):
        loc_x = 0
        for i in range(0, H_pad - HH + 1, stride):
            loc_y = 0
            for j in range(0, W_pad - WW + 1, stride):
                x_part = x_pad[index, :, i:i + HH, j:j + WW]  # C*HH*WW
                x_cov = x_part * w  # F*C*HH*WW
                x_cov = x_cov.sum(axis=(1, 2, 3))
                x_cov += b
                out[index, :, loc_x, loc_y] = x_cov
                loc_y += 1
            loc_x += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache  # x：输入图片，w, b：权重和偏置，conv_param：存放有pad和stride
    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = x
    N = x.shape[0]  # 图片数量
    F = w.shape[0]  # 滤波器数量
    C = w.shape[1]  # 通道数量

    H = x.shape[2]  # 输入 x 的高度
    W = x.shape[3]  # 输入 x 的宽度

    HH = w.shape[2]  # 滤波器的高度
    WW = w.shape[3]  # 滤波器的宽度

    # 是否需要填充？
    if pad != 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                       'constant')  # 对每一个通道进行填充

    H_pad = x_pad.shape[2]
    W_pad = x_pad.shape[3]

    H_out = 1 + (H + 2 * pad - HH) // stride  # 输出特征图的高度
    W_out = 1 + (W + 2 * pad - WW) // stride  # 输出特征图的宽度

    out = np.zeros((N, F, H_out, W_out))  # 初始化特征图
    dx = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    loc_x = 0
    loc_y = 0
    for index in range(N):
        for f in range(F):
            loc_x = 0
            for i in range(0, H_pad - HH + 1, stride):
                loc_y = 0
                for j in range(0, W_pad - WW + 1, stride):
                    # 计算 dw，注意不要漏掉了f，是按照f一个个求的
                    x_part = x_pad[index, :, i:i + HH, j:j + WW]  # C*HH*WW
                    dw[f] += dout[index, f, loc_x, loc_y] * x_part

                    # 计算dx
                    dx[index, :, i:i + HH,
                       j:j + WW] += w[f] * dout[index, f, loc_x, loc_y]

                    # 计算db,直接锁定那个值
                    db[f] += np.sum(dout[index, f, loc_x, loc_y])
                    loc_y += 1
                loc_x += 1
    dx = dx[:, :, pad:pad + H, pad:pad + W]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    无需填充
    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]  # 图片数量
    C = x.shape[1]  # 图片通道数
    H = x.shape[2]  # 图片高度
    W = x.shape[3]  # 图片宽度

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))
    # for index in range(N):
    #     loc_x = 0
    #     for i in range(0, H - pool_height + 1, stride):
    #         loc_y = 0
    #         for j in range(0, W - pool_width + 1, stride):
    #             max = np.max(x[index, :, i:i + pool_height, j:j + pool_width],axis=(1,2))  # C*HH*WW
    #             out[index, :, loc_x, loc_y] = max
    #             loc_y += 1
    #         loc_x += 1
    # 👇更优雅~

    for i in range(N):
        for j in range(C):
            for m in range(H_out):
                for n in range(W_out):
                    max = np.max(x[i, j, m * stride:m * stride + pool_height,
                                   n * stride:n * stride + pool_width])
                    out[i, j, m, n] = max

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N = x.shape[0]  # 图片数量
    C = x.shape[1]  # 图片通道数
    H = x.shape[2]  # 图片高度
    W = x.shape[3]  # 图片宽度
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    N, C, H_out, W_out = dout.shape
    dx = np.zeros_like(x)

    for i in range(N):
        for j in range(C):
            for m in range(H_out):
                for n in range(W_out):
                    index = np.argmax(x[i, j,
                                        m * stride:m * stride + pool_height,
                                        n * stride:n * stride +
                                        pool_width])  #第i张图片，第c通道，抓出一个二位数组（图像）
                    ind = np.unravel_index(index,
                                           (pool_height, pool_width))  # 坐标还原
                    dx[i, j, m * stride:m * stride + pool_height,
                       n * stride:n * stride + pool_width][ind] = dout[i, j, m,
                                                                       n]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape  #原本为(N,C,H,W)
    x_t = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)  # 转换为(N,H,W,C)
    out, cache = batchnorm_forward(x_t, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # 恢复为(N,C,H,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_t = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_t, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    num_c = C // G  # 每一组的通道数
    x_t = x.reshape(N, G, num_c, H, W)  # 将其分为 G 组，每组有 num_C 个通道，
    mean = np.mean(x_t, axis=(2, 3, 4), keepdims=True)  # 每一组求平均
    var = np.var(x_t, axis=(2, 3, 4), keepdims=True)  # 每一组求方差
    x_t = (x_t - mean) / np.sqrt(var + eps)  # 每一组正则化
    x_norm = x_t.reshape(N, C, H, W)
    out = x_norm * gamma + beta
    cache = (x, x_norm, gamma, mean, var, eps, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_norm, gamma, mean, var, eps, G = cache
    N, C, H, W = dout.shape
    M = C // G * H * W

    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))[None, :, None,
                                                   None]  # 这里得到了一个向量，需要把他还原回四维
    dbeta = np.sum(dout, axis=(0, 2, 3))[None, :, None, None]

    x_t = x.reshape(N, G, C // G, H, W)  # 变换形状，进行分组

    # (N,C,H,W)->(N,G,num_c,H,W) 当成[N,G]看，带入ln层
    dx_norm = (dout * gamma).reshape(N, G, C // G, H, W)

    dvar = np.sum(-0.5 * dx_norm * (x_t - mean) * ((var + eps)**-1.5),
                  axis=(2, 3, 4))

    dmean = np.sum(-dx_norm / np.sqrt(var + eps), axis=(2, 3, 4))
    dx = dx_norm / np.sqrt(var + eps) + dmean.reshape(
        N, G, 1, 1, 1) / M + dvar.reshape(N, G, 1, 1, 1) * 2 / M * (x_t - mean)
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
