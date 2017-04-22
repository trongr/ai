#!/usr/bin/env python

import tensorflow as tf

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant([ 
    [[1,2,3],[2,3,4],[3,4,5]], 
    [[4,5,6],[5,6,7],[6,7,8]], 
    [[7,8,9],[8,9,10],[9,10,11]] 
])

with tf.Session() as session:
    result = session.run(Scalar)
    print "Scalar (1 entry):\n %s \n" % result
    result = session.run(Vector)
    print "Vector (3 entries) :\n %s \n" % result
    result = session.run(Matrix)
    print "Matrix (3x3 entries):\n %s \n" % result
    result = session.run(Tensor)
    print "Tensor (3x3x3 entries) :\n %s \n" % result

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two
arraysAreEqual = tf.reduce_all(tf.equal(first_operation, second_operation, name=None))

with tf.Session() as session:
    result1 = session.run(first_operation)
    print "Defined using tensorflow function :"
    print(result1)
    result2 = session.run(second_operation)
    print "Defined using normal expressions :"
    print(result2)

    arraysAreEqualResult = session.run(arraysAreEqual)

    print("Are these two arrays the same?", arraysAreEqualResult)