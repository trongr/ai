#!/usr/bin/env python

import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = a + b

with tf.Session() as session:
    result = session.run(c)
    print(result)