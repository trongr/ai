from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import ptb.reader as reader

# Initial weight scale
init_scale = 0.1
# Initial learning rate
learning_rate = 1.0
# Maximum permissible norm for the gradient (For gradient clipping --
# another measure against Exploding Gradients)
max_grad_norm = 5
# The number of layers in our model
num_layers = 2
# The total number of recurrence steps, also known as the number of layers
# when our RNN is "unfolded"
num_steps = 20
# The number of processing units (neurons) in the hidden layers
hidden_size = 200
# The maximum number of epochs trained with the initial learning rate
max_epoch = 4
# The total number of epochs in training
max_max_epoch = 13
# The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
# At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
# The decay for the learning rate
decay = 0.5
# The size for each batch of data
batch_size = 30
# The size of our vocabulary
vocab_size = 10000
# Training flag to separate training from testing
is_training = 1
# Data directory for our dataset
data_dir = "./simple-examples/data/"

session = tf.InteractiveSession()

# Reads the data and separates it into training data, validation data and
# testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

iterator = reader.ptb_iterator(train_data, batch_size, num_steps)
first_tuple = iterator.next()
x = first_tuple[0]
y = first_tuple[1]

# # every row in x is a sentence, and each row in y is the same sentence shifted
# # by one into the future, i.e. the last word in each row in y is the next word
# # after the same row in x
# print "x shape", x.shape
# print "first 3 x sentences", x[0:3]
# print "y shape", y.shape
# print "first 3 y sentences", y[0:3]

size = hidden_size
_input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30, 20]
_targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30, 20]
feed_dict = {_input_data: x, _targets: y}
print "_input_data", session.run(_input_data, feed_dict)
