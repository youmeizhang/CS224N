#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:38:45 2019

@author: yumi.zhang
"""

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.
    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.
    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    
    #raise NotImplementedError
    ### END YOUR CODE
    
    h = sigmoid(X.dot(W1) + b1)
    y_h = softmax(h.dot(W2) + b2)
    
    ### YOUR CODE HERE: backward propagation

    cost = np.sum(-np.log(y_h[labels == 1])) / X.shape[0]
    
    delta1 = (y_h - labels) / X.shape[0]
    delta2 = delta1.dot(W2.transpose())
    delta3 = sigmoid_grad(h) * delta2
    
    #calculate gradient
    gradW1 = X.transpose().dot(delta3)
    gradb1 = np.sum(delta3, 0)
    
    gradW2 = h.transpose().dot(delta1)
    gradb2 = np.sum(delta1, 0, keepdims = True)
    
    #raise NotImplementedError
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    #total number of parameters in the network: (Dx + 1) * H + (H + 1) * Dy
    #because Dx * H + 1 * H = (Dx + 1) * H
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    #gradcheck_naive(f, x)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    print("Not Implemented Error")
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()