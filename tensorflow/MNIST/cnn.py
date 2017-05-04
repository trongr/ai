'''
Architecture:

    (Input) -> [BATCH_SIZE, 28, 28, 1]
    (Convolutional layer 1) -> [BATCH_SIZE, 28, 28, 32]
    (ReLU 1) -> [BATCH_SIZE, 28, 28, 32]
    (Max pooling 1) -> [BATCH_SIZE, 14, 14, 32]

    (Convolutional layer 2) -> [BATCH_SIZE, 14, 14, 64]
    (ReLU 2) -> [BATCH_SIZE, 14, 14, 64]
    (Max pooling 2) -> [BATCH_SIZE, 7, 7, 64]

    [fully connected layer 3] -> [BATCH_SIZE, 1024]
    [ReLU 3] -> [BATCH_SIZE, 1024]
    [Drop out] -> [BATCH_SIZE, 1024]
    [fully connected layer 4] -> [BATCH_SIZE, 10]
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28  # width of the image in pixels
height = 28  # height of the image in pixels
flat = width * height  # number of pixels in one image
class_output = 10  # number of possible classifications for the problem

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolutional layer 1

# 32 5x5-filters
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[
                         1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[
                         1, 2, 2, 1], padding='SAME')  # max_pool_2x2
layer1 = h_pool1

# convolutional layer 2

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

convolve2 = tf.nn.conv2d(layer1, W_conv2, strides=[
                         1, 1, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[
                         1, 2, 2, 1], padding='SAME')  # max_pool_2x2
layer2 = h_pool2
layer2_matrix = tf.reshape(layer2, [-1, 7 * 7 * 64])

# fully connected layer 3

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fcl3 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl3)
layer3 = h_fc1

# dropout

keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)

# class probabilities

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

fcl4 = tf.matmul(layer3_drop, W_fc2) + b_fc2
layer4 = tf.nn.softmax(fcl4)

# loss function
cross_entropy = - \
    tf.reduce_mean(tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(layer4, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):  # Load 50 training examples for each training iteration
        batch = mnist.train.next_batch(50)  # batch = (images, labels)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={
                           x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i % 10 == 0:
            print("Iteration {}. loss: {}".format(i, loss))

    acc = accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5}) * 100
    print("Final accuracy: {}% ".format(acc))
