"""
python charrnn.py shakespeare.txt
"""

import sys
import numpy as np
from random import shuffle
import tensorflow as tf

FILENAME = sys.argv[1]
DATA = open(FILENAME, 'r').read()
DATA = DATA[:100000]  # poij toggle truncate data
CHARS = list(set(DATA))
DATA_SIZE, VOCAB_SIZE = len(DATA), len(CHARS)
CHAR_TO_IX = {ch: i for i, ch in enumerate(CHARS)}
IX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}

# print "DATA", DATA
# print 'DATA has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE)
# print "CHARS", CHARS
# print "CHAR_TO_IX", CHAR_TO_IX
# print "IX_TO_CHAR", IX_TO_CHAR

BATCH_SIZE = 100
SEQ_LENGTH = 20
NUM_LSTM_CELLS = 3
NUM_CLASSES = len(CHARS)

"""
Generate training input and convert into an array of shape (None, SEQ_LENGTH,
1):

dataX = [
    [[1], [0], [1], [1],...],
    [[0], [0], [0], [1],...],
    [[1], [1], [1], [1],...],
    ...
] ~ (None, SEQ_LENGTH, 1)

Generate training output and convert into one-hot array of shape (None,
NUM_CLASSES):

dataY = [
    [0,..., 0, 1, 0,..., 0],
    [1,..., 0, 0, 0,..., 0],
    [0,..., 0, 0, 0,..., 1],
    ...
] ~ (None, NUM_CLASSES)
"""


def genData():
    dataX = []
    dataY = []

    NUM_DATA = len(DATA) - SEQ_LENGTH
    idx = range(NUM_DATA)
    shuffle(idx)
    for i in idx:
        x = [[CHAR_TO_IX[ch]] for ch in list(DATA[i:i + SEQ_LENGTH])]
        y = CHAR_TO_IX[DATA[i + SEQ_LENGTH]]
        dataX.append(x)
        dataY.append(y)

    # Converting dataY values into one-hot
    zeros = np.zeros((len(dataY), NUM_CLASSES))
    zeros[np.arange(len(dataY)), dataY] = 1.0
    dataY = zeros

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY


dataX, dataY = genData()

"""
Separating training and test data.
"""
NUM_TRAIN = int(0.8 * len(dataX))
trainX = dataX[:NUM_TRAIN]
trainY = dataY[:NUM_TRAIN]
testX = dataX[NUM_TRAIN:]
testY = dataY[NUM_TRAIN:]

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
RNN layer
"""
NUM_CELL_UNITS = 25  # poij. Adjust hyperparam
Cell = tf.contrib.rnn.BasicLSTMCell(NUM_CELL_UNITS, state_is_tuple=True)
Cells = tf.contrib.rnn.MultiRNNCell([Cell] * NUM_LSTM_CELLS)
Output, State = tf.nn.dynamic_rnn(Cells, X, dtype=tf.float32)
Output = tf.reshape(Output, [tf.shape(Output)[0], -1])

"""
Fully connected layer mapping RNN output to classes
"""
W = tf.Variable(tf.truncated_normal(
    [SEQ_LENGTH * NUM_CELL_UNITS, NUM_CLASSES]))
b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
Scores = tf.matmul(Output, W) + b

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

EPOCHS = 50
NUM_BATCHES = int(NUM_TRAIN / BATCH_SIZE)
for i in range(EPOCHS):
    print "EPOCH", i
    ptr = 0
    for j in range(NUM_BATCHES):
        batchX = trainX[ptr:ptr + BATCH_SIZE]
        batchY = trainY[ptr:ptr + BATCH_SIZE]
        minimize, output, state = sess.run(
            [Minimize, Output, State], {X: batchX, Y: batchY})

        if j % 10 == 0:
            accuracy = sess.run(Accuracy, {X: batchX, Y: batchY})
            print('Batch: {:2d}, accuracy: {:3.1f} %'.format(
                j, 100 * accuracy))

        ptr += BATCH_SIZE

"""
TESTING
"""
accuracy = sess.run(Accuracy, {X: testX, Y: testY})
print('Accuracy on test set: {:3.1f} %'.format(100 * accuracy))

# Don't need to pass testY in here because Pred doesn't need it:
pred = sess.run(Pred, {X: testX})
print 'Test case:', testX[:1]
print 'Prediction:', pred[:1]

sess.close()
