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
  # print(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  for i in range(score.shape[0]):
    # dScore = np.zeros_like(score[i])
    mx = np.max(score[i,:])
    score[i,:] -= mx
    
    de = np.sum(np.exp(score[i,:]))
    nm = np.exp(score[i,y[i]])
    loss += -np.log(nm/de)
    dScore = np.exp(score[i,:])/de
    dScore[y[i]] = -(1-nm/de)
    # print(dScore)
    dW += X[i].reshape((-1,1)).dot(dScore.reshape((1,-1)))
  
  loss /= score.shape[0]
  loss += reg*np.sum(W*W)

  dW /= score.shape[0]
  dW += reg*2*W
  # print(loss, dW)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  score = X.dot(W)
  mx = np.max(score, axis=1)
  score -= mx.reshape((-1,1))
  ds = np.exp(score)
  ns = ds[range(X.shape[0]),y]
  loss = np.mean(-np.log(ns/np.sum(ds, axis=1)))
    
  dScore = ds/np.sum(ds, axis=1).reshape((-1,1))
  dScore[range(X.shape[0]),y] = ns/np.sum(ds, axis=1)-1
    
  dW = X.T.dot(dScore)
  dW /= X.shape[0]
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

