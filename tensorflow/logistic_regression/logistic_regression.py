#!/usr/bin/env python
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

# iris.data looks like [[5.1  3.5  1.4  0.2],...] where the quadruples are the
# lengths and the widths of the sepals and petals, in centimetres
print("iris.data--------------------")
print(iris.data)
print "iris.data length:", len(iris.data)

# iris.target looks like [0, 1, 2,...] where the numbers indicate the types of
# Iris: 0, 1, 2 for Iris setosa, Iris virginica and Iris versicolor, resp.
print("iris.target------------------")
print(iris.target)
print "iris.target length:", len(iris.target)

iris_X, iris_y = iris.data[:, :], iris.target[:]
print "iris_X:", iris_X
print "iris_X length:", len(iris_X)
print "iris_y:", iris_y
print "iris_y length:", len(iris_y)

# one hot encoding of iris_y target classes
iris_y = pd.get_dummies(iris_y).values
print "iris_y dummies:", iris_y
print "iris_y dummies length:", len(iris_y)

# random_state guarantees deterministic random split
trainX, testX, trainY, testY = train_test_split(
    iris_X, iris_y, test_size=0.33, random_state=42)
print "trainX", trainX
print "trainX length", len(trainX)
print "trainY", trainY
print "trainY length", len(trainY)
print "testX", testX
print "testX length", len(testX)
print "testY", testY
print "testY length", len(testY)

# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.
print "trainX.shape", trainX.shape
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
# In the iris dataset, this number is '3'.
print "trainY.shape", trainY.shape
numLabels = trainY.shape[1]

# Iris has 4 features, so X is a tensor to hold our data.
X = tf.placeholder(tf.float32, [None, numFeatures])
# This will be our correct answers matrix for 3 classes.
y = tf.placeholder(tf.float32, [None, numLabels])

# Randomly sample from a normal distribution with standard deviation .01
weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))
bias = tf.Variable(tf.random_normal([1, numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

cost_OP = tf.nn.l2_loss(activation_OP - y, name="squared_error_cost")
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
init_OP = tf.global_variables_initializer()

# argmax(activation_OP, 1) returns the label with the most probability
# argmax(y, 1) is the correct label
correct_predictions_OP = tf.equal(
    tf.argmax(activation_OP, 1), tf.argmax(y, 1))

# If every false prediction is 0 and every true prediction is 1, the average
# returns us the accuracy (percentage of accurate predictions)
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
activation_summary_OP = tf.summary.histogram("output", activation_OP)
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP])

cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

with tf.Session() as sess:
    sess.run(init_OP)
    for i in range(700):
        if i > 1 and diff < .0001:
            print("change in cost %g; convergence." % diff)
            break
        else:
            step = sess.run(training_OP, feed_dict={X: trainX, y: trainY})
        if i % 10 == 0:
            epoch_values.append(i)
            train_accuracy, newCost = sess.run(
                [accuracy_OP, cost_OP], feed_dict={X: trainX, y: trainY})
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)
            diff = abs(newCost - cost)
            cost = newCost
            print("step %d, training accuracy %g, cost %g, change in cost %g" %
                  (i, train_accuracy, newCost, diff))

    print("final accuracy on test set: %s" %
          str(sess.run(accuracy_OP, feed_dict={X: testX, y: testY})))

    writer = tf.summary.FileWriter("summary_logs", sess.graph)

plt.plot(cost_values)
plt.show()

plt.plot(accuracy_values)
plt.show()
