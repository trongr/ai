import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D

mnist = input_data.read_data_sets(".", one_hot=True)

trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels

print "Train Images: ", trainimgs.shape
print "Train Labels  ", trainlabels.shape
print "Test Images:  ", testimgs.shape
print "Test Labels:  ", testlabels.shape
print
print "Train label 1", trainlabels[0]

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]

print "ntrain", ntrain
print "ntest", ntest
print "dim", dim
print "nclasses", nclasses

# # You can change these numbers here to see other samples
# samplesIdx = [100, 101, 102]

# fig = plt.figure()

# ax1 = fig.add_subplot(121)
# ax1.imshow(testimgs[samplesIdx[0]].reshape([28, 28]), cmap='gray')

# xx, yy = np.meshgrid(np.linspace(0, 28, 28), np.linspace(0, 28, 28))
# X = xx
# Y = yy
# Z = 100 * np.ones(X.shape)

# img = testimgs[77].reshape([28, 28])
# ax = fig.add_subplot(122, projection='3d')
# ax.set_zlim((0, 200))

# offset = 200
# for i in samplesIdx:
#     img = testimgs[i].reshape([28, 28]).transpose()
#     ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap="gray")
#     offset -= 100

#     ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# plt.show()

# for i in samplesIdx:
# print "Sample: {0} - Class: {1} - Label Vector: {2} ".format(i,
# np.nonzero(testlabels[i])[0], testlabels[i])

n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x")
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)
    # The output of the rnn would be a [100x28x128] matrix. we use the linear
    # weights to map it to a [?x10 matrix] activation
    output = tf.reshape(tf.split(outputs, 28, axis=1,
                                 num=None, name='split')[-1], [-1, 128])
    return tf.matmul(output, weights['out']) + biases['out']


with tf.variable_scope('forward3'):
    pred = RNN(x, weights, biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        # We will read a batch of 100 images [100 x 784] as batch_x batch_y is a
        # matrix of [100x10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # We consider each row of the image as one sequence Reshape data to get
        # 28 seq of 28 elements, so that, batch_x is [100x28x28]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
