from builtins import range
from builtins import object
from operator import imod
from typing import Counter
from matplotlib.pyplot import close
import numpy as np
from past.builtins import xrange
import collections
import math


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        输入 X 为（500，3072）包含的是测试图像

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
          返回距离数组dists（测试图像数，训练图象数），dists[i,j]表示第i个测试图像和第j个测试图像之间的距离
        """
        num_test = X.shape[0]   # shape[0]代表数组的行数，shape[1]代表的是矩阵的列数
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                # dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
                # 错误原因，是平方求和，不是求和平方
                # dists[i][j] = np.sqrt(pow(np.sum(self.X_train[j]-X[i]), 2))
                dists[i][j] = np.sqrt(np.sum((self.X_train[j]-X[i])**2))
                # np.sum()：用做与向量求和，可以按照行或者列求和
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i] = np.sqrt(np.sum((self.X_train-X[i])**2, axis=1))
            # print(np.shape(np.sqrt(np.sum((self.X_train-X[i])**2, axis=1))))
            # 根据numpy数组的广播特性，直接进行向量计算
            # 注意axis=1，得到一个5000x1的数组
            # 奇怪了所以在dists[i]=。。。的地方这个向量自己转置成列向量了？
            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # print(np.shape(dists))  # (500,5000)
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X_train_trans = np.transpose(self.X_train)
        # 这里就是把平方和公式展开即可
        temp_2ab = np.dot(X, self.X_train.T)*2
        temp_a2 = np.sum(np.square(X), axis=1, keepdims=True)
        temp_b2 = np.sum(np.square(self.X_train), axis=1)
        dists = temp_a2 - temp_2ab + temp_b2
        dists = np.sqrt(dists)
        # print(np.shape(temp_2ab))
        # print(np.shape(temp_a2))
        # print(np.shape(temp_b2))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
          y：一个numpy数组，大小为(测试集大小)，包含了预测的标签，y[i]代表对于该测试图像的预测
        """
        num_test = dists.shape[0]  # 测试集大小，500
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
           #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            count = k
            dists_sort = np.argsort(dists[i])  # 对距离进行排序,返回的是索引
            for j in range(k):
                closest_y.append(self.y_train[dists_sort[j]])

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            label = list(set(closest_y))
            lable_num = [closest_y.count(l)for l in label]
            y_pred[i] = label[np.argmax(lable_num)]

            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################
        return y_pred
