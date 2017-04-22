#!/usr/bin/env python
import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.add(a, b)
d = tf.subtract(a, b)

with tf.Session() as session:
    result = session.run(c)
    print 'c == %s' % result
    result = session.run(d)
    print 'd == %s' % result
