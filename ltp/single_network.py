import numpy as np
from scipy.special import softmax
import random

def cross_entropy(y_pred, y_label):
    assert y_pred.shape == y_label.shape
    sample_num = y_label.shape[0]
    s = softmax(y_pred, 1)
    loss = - np.sum(np.multiply(y_label, np.log(s))) / sample_num
    return loss

def grad_cross_entropy(y_pred, y_label):
    sample_num = y_label.shape[0]
    grad = softmax(y_pred, 1)
    grad -= y_label
    grad = grad / sample_num
    return grad

def sample(indexs, data):
    out = []
    for i in indexs:
        out.append(data[i])
    return np.array(out)

