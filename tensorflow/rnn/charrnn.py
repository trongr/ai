"""
python charrnn.py shakespeare.txt
"""

import sys
import numpy as np
import random
import string
import tensorflow as tf

FILENAME = sys.argv[1]
DATA = open(FILENAME, 'r').read()
# DATA = DATA[100000:110000]
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
SEQ_LENGTH = 100
NUM_CELL_UNITS = 256
NUM_LSTM_CELLS = 2
NUM_CLASSES = len(CHARS)

"""
Generate training batches
"""
SIZE_DATA = len(DATA) - SEQ_LENGTH - 1 # - SEQ_LENGTH - 1 to avoid clipping (short strings) near the end
NUM_TRAIN = int(0.8 * SIZE_DATA) # 80-20 train-test split
batch_ptr = 0 # pointer to start char in DATA for current batch

def nextTrainBatch():
    global batch_ptr
    batchX = []
    batchY = []
    for i in xrange(BATCH_SIZE):
        x = [CHAR_TO_IX[ch] for ch in list(DATA[batch_ptr:batch_ptr + SEQ_LENGTH])]
        y = CHAR_TO_IX[DATA[batch_ptr + SEQ_LENGTH]] 
        batchX.append(x)
        batchY.append(y)
        batch_ptr = (batch_ptr + 1) % NUM_TRAIN # loop around for further training
    batchX = np.array(batchX)
    batchY = np.array(batchY)
    return batchX, batchY

def genTestXFromString(s):
    ixes = [[CHAR_TO_IX[ch] for ch in s]]
    return ixes

def sample(testX):
    output = []    
    s = ixes_to_string(testX[0])
    GEN_STR_LEN = 300

    print "TRAINING"
    print "--------"
    print ixes_to_string(testX[0])
    print "--------"

    for i in xrange(GEN_STR_LEN):
        testX = genTestXFromString(s)
        predictions = sess.run(Predictions, {X: testX})
        char = char_distr_to_char(predictions)
        output.append(char)
        s = (s + char)[-SEQ_LENGTH:]

    print "SAMPLE"
    print "======"
    print "".join(output)
    print "======"

def ixes_to_string(ixes):
    """
    Convert a list of indices to a string
    """
    return "".join([IX_TO_CHAR[i] for i in ixes])

def char_distr_to_char_ix(predictions):
    """
    Prediction distributions for a single char, i.e. the probability of a
    character being correct is given by the corresponding probability in
    predictions. NOTE. The correct char is not the one with the highest
    probability: np.argmax(predictions) is not the character you're looking for.
    Instead we choose a char index out of range(NUM_CLASSES) based on
    probabilities given by predictions.
    """
    return np.random.choice(range(NUM_CLASSES), p=predictions.ravel())

def char_distr_to_char(predictions):
    return IX_TO_CHAR[char_distr_to_char_ix(predictions)]

def print_test_results(testX, predictions):
    for i in xrange(len(testX)):
        test_str = testX[i].reshape(-1)
        char_distr = predictions[i].reshape(-1)
        predicted_char = char_distr_to_char(char_distr)
        print ixes_to_string(test_str), "\t", predicted_char

"""
Placeholders for minibatch input and output data
"""
X = tf.placeholder(tf.int32, [None, SEQ_LENGTH]) # the sequences
Y = tf.placeholder(tf.int32, [None]) 
X_onehot = tf.one_hot(X, NUM_CLASSES)
Y_onehot = tf.one_hot(Y, NUM_CLASSES)

"""
BUILDING THE GRAPH
BUILDING THE GRAPH
BUILDING THE GRAPH
"""

# RNN layer
Cell = tf.contrib.rnn.BasicLSTMCell(NUM_CELL_UNITS, state_is_tuple=True)
Cells = tf.contrib.rnn.MultiRNNCell([Cell] * NUM_LSTM_CELLS)
InitState = Cells.zero_state(tf.shape(X)[0], tf.float32)
Output, State = tf.nn.dynamic_rnn(Cells, X_onehot, initial_state=InitState, dtype=tf.float32)

# Fully connected layer mapping RNN output to classes
W = tf.Variable(tf.truncated_normal([NUM_CELL_UNITS, NUM_CLASSES]), dtype=tf.float32)
b = tf.Variable(tf.constant(0, shape=[NUM_CLASSES], dtype=tf.float32))
Last = tf.gather(tf.transpose(Output, [1, 0, 2]), SEQ_LENGTH - 1)
Logits = tf.matmul(Last, W) + b

# Results
Predictions = tf.nn.softmax(Logits)
Corrects = tf.equal(tf.argmax(Y_onehot, axis=1), tf.argmax(Predictions, axis=1))
Accuracy = tf.reduce_mean(tf.cast(Corrects, tf.float32))

"""
Loss and optimization
"""
Losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Logits)
Loss = tf.reduce_mean(Losses)
LearningRate = tf.train.exponential_decay(learning_rate=1e-1,
    global_step=1, decay_steps=NUM_TRAIN, decay_rate=0.95,
    staircase=True)
Minimize = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

"""
TRAINING
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batchX, _ = nextTrainBatch()
state = sess.run(InitState, {X: batchX})

i = 0
while True: 
    batchX, batchY = nextTrainBatch()
    minimize, loss, state = sess.run([
        Minimize, Loss, State
    ], {
        X: batchX, Y: batchY, InitState: state
    })
    if i % 50 == 0:
        accuracy = sess.run(Accuracy, {X: batchX, Y: batchY})
        print('Batch: {:2d}, loss: {:.4f}, accuracy: {:3.1f} %'
            .format(i, loss, 100 * accuracy))
        sample([batchX[0]])

    i += 1

# """
# TESTING. TODO. generate testX testY from DATA similar to nextTrainBatch()
# """
# accuracy = sess.run(Accuracy, {X: testX, Y: testY})
# print('Accuracy on test set: {:3.1f} %'.format(100 * accuracy))

# # Don't need to pass testY in here because Predictions doesn't need it:
# predictions = sess.run(Predictions, {X: testX})
# print_test_results(testX, predictions)

sess.close()
