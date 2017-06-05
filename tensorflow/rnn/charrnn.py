"""
python charrnn.py shakespeare.txt
"""

import sys
import numpy as np
from random import shuffle
import tensorflow as tf

FILENAME = sys.argv[1]
DATA = open(FILENAME, 'r').read()
CHARS = list(set(DATA))
DATA_SIZE, VOCAB_SIZE = len(DATA), len(CHARS)
CHAR_TO_IX = {ch: i for i, ch in enumerate(CHARS)}
IX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}

# print "DATA", DATA
print 'DATA has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE)
print "CHARS", CHARS
print "CHAR_TO_IX", CHAR_TO_IX
print "IX_TO_CHAR", IX_TO_CHAR

BATCH_SIZE = 100
SEQ_LENGTH = 10
NUM_CELL_UNITS = 1024
NUM_LSTM_CELLS = 3
NUM_CLASSES = len(CHARS)

"""
Generate training input and convert into an array of shape (None, SEQ_LENGTH,
1):

batchX = [
    [[1], [0], [1], [1],...],
    [[0], [0], [0], [1],...],
    [[1], [1], [1], [1],...],
    ...
] ~ (None, SEQ_LENGTH, 1)

Generate training output and convert into one-hot array of shape (None,
NUM_CLASSES):

batchY = [
    [0,..., 0, 1, 0,..., 0],
    [1,..., 0, 0, 0,..., 0],
    [0,..., 0, 0, 0,..., 1],
    ...
] ~ (None, NUM_CLASSES)
"""
SIZE_DATA = len(DATA) - SEQ_LENGTH # - SEQ_LENGTH to avoid clipping (short strings) near the end
NUM_TRAIN = int(0.8 * SIZE_DATA) # 80-20 train-test split
batch_ptr = 0 # pointer to start char in DATA for current batch
def nextTrainBatch():
    global batch_ptr
    batchX = []
    batchY = []
    for i in xrange(BATCH_SIZE):
        x = [[CHAR_TO_IX[ch]] for ch in list(DATA[batch_ptr:batch_ptr + SEQ_LENGTH])]
        y = CHAR_TO_IX[DATA[batch_ptr + SEQ_LENGTH]]
        batchX.append(x)
        batchY.append(y)
        batch_ptr = (batch_ptr + 1) % NUM_TRAIN # loop around for further training    

    # Converting dataY values into one-hot
    zeros = np.zeros((len(batchY), NUM_CLASSES))
    zeros[np.arange(len(batchY)), batchY] = 1.0
    batchY = zeros

    batchX = np.array(batchX)
    batchY = np.array(batchY)

    return batchX, batchY

def genTestXFromString(s):
    ixes = [[[CHAR_TO_IX[ch]] for ch in s]]
    return ixes

def sample():
    s = "To be, or not to be- that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune Or to take arms against a sea of troubles, And by opposing end them."
    s = s[:SEQ_LENGTH] # make it fit our model's input
    output = []
    GEN_STR_LEN = 300
    for i in xrange(GEN_STR_LEN):
        testX = genTestXFromString(s)
        pred = sess.run(Pred, {X: testX})
        char = char_distr_to_char(pred)
        s = (s + char)[-SEQ_LENGTH:]
        output.append(char)

    print "=============================================="
    print "".join(output)
    print "=============================================="

def ixes_to_string(ixes):
    """
    Convert a list of indices to a string
    """
    return "".join([IX_TO_CHAR[i] for i in ixes])

def char_distr_to_char_ix(pred):
    """
    Prediction distribution for a single char, i.e. the argument with the max
    value is the predicted char.
    """
    return np.argmax(pred)

def char_distr_to_char(pred):
    return IX_TO_CHAR[char_distr_to_char_ix(pred)]

def print_test_results(testX, pred):
    for i in xrange(len(testX)):
        test_str = testX[i].reshape(-1)
        char_distr = pred[i].reshape(-1)
        predicted_char = char_distr_to_char(char_distr)
        print ixes_to_string(test_str), "\t", predicted_char

"""
Placeholders for minibatch input and output data
"""
X = tf.placeholder(tf.float32, [None, SEQ_LENGTH, 1])
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

"""
BUILDING THE GRAPH
BUILDING THE GRAPH
BUILDING THE GRAPH
"""

"""
Pass all activations to FC layer
"""
"""
# RNN layer
Cell = tf.contrib.rnn.BasicLSTMCell(NUM_CELL_UNITS, state_is_tuple=True)
Cells = tf.contrib.rnn.MultiRNNCell([Cell] * NUM_LSTM_CELLS)
Output, State = tf.nn.dynamic_rnn(Cells, X, dtype=tf.float32)
Output = tf.reshape(Output, [tf.shape(Output)[0], -1]) 

# Fully connected layer mapping RNN output to classes
W = tf.Variable(tf.truncated_normal([SEQ_LENGTH * NUM_CELL_UNITS, NUM_CLASSES]))
b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
Scores = tf.matmul(Output, W) + b
"""

"""
Only pass the last activations
"""
# RNN layer
Cell = tf.contrib.rnn.BasicLSTMCell(NUM_CELL_UNITS, state_is_tuple=True)
Cells = tf.contrib.rnn.MultiRNNCell([Cell] * NUM_LSTM_CELLS)
Output, State = tf.nn.dynamic_rnn(Cells, X, dtype=tf.float32)
Last = tf.gather(tf.transpose(Output, [1, 0, 2]), SEQ_LENGTH - 1) 

# Fully connected layer mapping RNN output to classes
W = tf.Variable(tf.truncated_normal([NUM_CELL_UNITS, NUM_CLASSES]))
b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
Scores = tf.matmul(Last, W) + b

"""
Loss and optimization
"""
Pred = tf.nn.softmax(Scores)
Loss = - tf.reduce_mean(tf.reduce_sum(Y * tf.log(Pred)))  # cross entropy
Minimize = tf.train.AdamOptimizer().minimize(Loss)
Corrects = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Pred, axis=1))
Accuracy = tf.reduce_mean(tf.cast(Corrects, tf.float32))

"""
TRAINING
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
while True: 
    batchX, batchY = nextTrainBatch()
    minimize, loss, state = sess.run([Minimize, Loss, State], {X: batchX, Y: batchY})

    if i % 50 == 0:
        accuracy = sess.run(Accuracy, {X: batchX, Y: batchY})
        print('Batch: {:2d}, loss: {:.4f}, accuracy: {:3.1f} %'.format(i, loss, 100 * accuracy))
        sample()

    i += 1

# """
# TESTING. TODO. generate testX testY from DATA similar to nextTrainBatch()
# """
# accuracy = sess.run(Accuracy, {X: testX, Y: testY})
# print('Accuracy on test set: {:3.1f} %'.format(100 * accuracy))

# # Don't need to pass testY in here because Pred doesn't need it:
# pred = sess.run(Pred, {X: testX})
# print_test_results(testX, pred)

sess.close()
