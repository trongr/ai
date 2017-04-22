#!/usr/bin/env python

import tensorflow as tf

sum = tf.Variable(0)
increment = tf.constant(1)
newSum = sum + increment
update = tf.assign(sum, newSum) # assigns newSum to sum

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for _ in range(10):
        session.run(update)
        print("sum:", session.run(sum))