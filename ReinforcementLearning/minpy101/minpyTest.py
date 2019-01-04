import numpy as np
import mxnet as mx
from mxnet.contrib.autograd import grad_and_loss
from examples.utils.data_utils import gaussian_cluster_generator

# Predict the class using multinomial logistic regression (softmax regression).
def predict(w, x):
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    return prob


def train_loss(w, x):
    prob = predict(w, x)
    loss = -np.sum(label * np.log(prob)) / num_samples
    return loss


"""Use Minpy's auto-grad to derive a gradient function off loss"""
grad_function = grad_and_loss(train_loss)

# Using gradient descent to fit the correct classes.
def train(w, x, loops):
    for i in range(loops):
        dw, loss = grad_function(w, x)
        if i % 10 == 0:
            print("Iter {}, training loss {}".format(i, loss))
        # gradient descent
        w -= 0.1 * dw


# Initialize training data.
num_samples = 10000
num_features = 500
num_classes = 5
data, label = gaussian_cluster_generator(num_samples, num_features, num_classes)
data = mx.nd.array(data)

# Initialize training weight and train
weight = np.random.randn(num_features, num_classes)
weight = mx.nd.array(weight)
train(weight, data, 100)
