import numpy as np
import tensorflow as tf

sample_input = tf.constant([
    [1, 2, 3, 4, 3, 2, 10, 11, 12],
    [3, 2, 2, 2, 2, 2, 10, 11, 12]
], dtype=tf.float32)
LSTM_CELL_SIZE = 3  # 2 hidden nodes

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2, LSTM_CELL_SIZE]),) * 2

with tf.variable_scope("LSTM"):
    output, new_state = lstm_cell(sample_input, state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print "sample_input", sess.run(sample_input)
    print "state", sess.run(state)
    print "output", sess.run(output)
    print "new_state", sess.run(new_state)
