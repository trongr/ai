import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])  # 28 x 28 pixels
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 10 classes

W = tf.Variable(tf.zeros([784, 10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # activation (class probabilities)
cross_entropy_loss = - \
    tf.reduce_mean(tf.reduce_sum(
        y_ * tf.log(y), reduction_indices=[1]))  # loss function

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):  # Load 50 training examples for each training iteration
        batch = mnist.train.next_batch(50)  # batch = (images, labels)
        _, loss = sess.run([train_step, cross_entropy_loss], feed_dict={
            x: batch[0], y_: batch[1]})
        if i % 100 == 0:
            print("Iteration {}. loss: {}".format(i, loss))

    acc = accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
    print("Final accuracy: {}% ".format(acc))
