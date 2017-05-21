import numpy as np
import tensorflow as tf

# 3 hidden nodes (equal to number of time steps)
LSTM_CELL_SIZE = 3
sample_batch_size = 2
num_layers = 2  # Lstm layers
sample_input = tf.constant([
    [
        [1, 2, 3, 4, 3, 2],
        [1, 2, 1, 1, 1, 2],
        [1, 2, 2, 2, 2, 2]],
    [
        [1, 2, 3, 4, 3, 2],
        [3, 2, 2, 1, 1, 2],
        [0, 0, 0, 0, 3, 2]
    ]
], dtype=tf.float32)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
initial_state = stacked_lstm.zero_state(sample_batch_size, tf.float32)
with tf.variable_scope("Stacked_LSTM_sample8"):
    output, new_state = tf.nn.dynamic_rnn(
        stacked_lstm, sample_input, dtype=tf.float32, initial_state=initial_state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print "sample_input", sess.run(sample_input)
    print "initial_state", sess.run(initial_state)
    print "new_state", sess.run(new_state)
    print "output", sess.run(output)
