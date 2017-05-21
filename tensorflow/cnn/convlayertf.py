import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def conv2d(X, W):
    return tf.nn.conv2d(input=X, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def MaxPool(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


im = misc.imread("./lena.png").astype(np.float)
grayim = np.dot(im[..., :3], [0.299, 0.587, 0.114])
print "Original image shape", grayim.shape

Image = np.expand_dims(np.expand_dims(grayim, 0), -1)
print "Expanded image shape", Image.shape

img = tf.placeholder(tf.float32, [None, 512, 512, 1])
print "Image placeholder", img.get_shape().as_list()

weights = {
    # 5 x 5 convolution, 1 input image, 32 outputs
    'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32]))
}

biases = {
    'b_conv1': tf.Variable(tf.random_normal([32]))
}

conv1 = tf.nn.relu(conv2d(img, weights['W_conv1']) + biases['b_conv1'])
maxpool1 = MaxPool(conv1)
print "conv1 shape", conv1.get_shape().as_list()
print "maxpool1 shape", maxpool1.get_shape().as_list()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_maxpool1 = sess.run(maxpool1, feed_dict={img: Image})
    print "result_maxpool1 shape", result_maxpool1.shape
    conv1_result = np.reshape(result_maxpool1, (256, 256, 32))
    for i in range(32):
        image = conv1_result[:, :, i]
        image *= 255.0 / image.max()
        plt.imshow(image, cmap=plt.get_cmap("gray"))
        plt.xlabel(i, fontsize=20, color='red')
        plt.show()
        plt.close()
