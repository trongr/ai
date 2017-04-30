import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
convValid = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
convSame = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    convValidResult = sess.run(convValid)
    convSameResult = sess.run(convSame)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with 'valid' padding (no padding)\n")
    print(convValidResult)
    print "conv valid shape", convValidResult.shape
    print('\n')
    print("Result/Feature Map with 'same' padding (with padding)\n")
    print(convSameResult)
    print "conv same shape", convSameResult.shape
