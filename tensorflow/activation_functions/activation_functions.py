import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def idfun(x):
    return x


def plot_act(i=1.0, actfun=lambda x: x):
    weights = np.arange(-10.0, 10.0, 1.0)
    biases = np.arange(-10.0, 10.0, 1.0)
    X, Y = np.meshgrid(weights, biases)

    os = np.array([actfun(tf.constant(i * w + b)).eval(session=sess)
                   for w, b in zip(np.ravel(X), np.ravel(Y))])
    Z = os.reshape(X.shape)

    # print "os", os
    # print "len(os)", len(os) # 400
    # print "Z", Z
    # print "len(Z), len(Z[0])", len(Z), len(Z[0]) # 20, 20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    plt.show()


with tf.Session() as sess:
    # plot_act(2.0, idfun)
    # plot_act(2.0, tf.sigmoid)
    # plot_act(2.0, tf.tanh)
    plot_act(2.0, tf.nn.relu)
