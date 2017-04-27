#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)

X = np.arange(0.0, 5.0, 0.1)

# # You can adjust the slope and intercept to verify the changes in the graph
# a = 1
# b = 0

# Y = a * X + b

# plt.plot(X, Y)
# plt.ylabel('Dependent Variable')
# plt.xlabel('Indepdendent Variable')
# plt.show()
# plt.close()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
# add a little random noise to every point y in y_data
y_data = np.vectorize(
    lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

# plt.scatter(x_data, y_data)
# plt.show()
# plt.close()

# a and b are weights we'll be optimizing
a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b  # y are our guesses

# diff between guesses y and ground truth y_data
loss = tf.reduce_mean(tf.square(y - y_data))

learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_data = []
    for step in range(100):
        # evals looks like [train_result, a_result, b_result]. train_result ==
        # None, cause sess.run(train) doesn't return anything. slice [1:] to get
        # [a_result, b_result].
        evals = sess.run([train, a, b])[1:]
        train_data.append(evals)
        if step % 5 == 0:
            print(step, evals)
            train_data.append(evals)

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    # make line colors start from yellow (1, 1, 0) and approach purple (1, 0, 1)
    # each iteration
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0:
        cb = 1.0
    if cg < 0.0:
        cg = 0.0

    # add line to plot
    [a, b] = f
    f_y = np.vectorize(lambda x: a * x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr, cg, cb))

plt.plot(x_data, y_data, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()
