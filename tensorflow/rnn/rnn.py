import time
import sys
import glob
import numpy as np
import random
import string
import tensorflow as tf

class RNN(object):

    def __init__(self, sess, DATA, NUM_CLASSES):
        """
        X is a list of integers from 0 to NUM_CLASSES - 1 inclusive, where NUM_CLASSES is the number
        of classes.
        """
        self.BATCH_SIZE = 100
        self.SEQ_LENGTH = 100
        self.NUM_CELL_UNITS = 64
        self.NUM_LSTM_CELLS = 5
        self.DATA = DATA
        self.NUM_CLASSES = NUM_CLASSES
        self.SIZE_DATA = len(self.DATA) - self.SEQ_LENGTH - 1 # - SEQ_LENGTH - 1 to avoid clipping (short strings) near the end
        self.NUM_TRAIN = int(0.8 * self.SIZE_DATA) # 80-20 train-test split
        self.batch_ptr = random.randint(0, self.NUM_TRAIN) # start training at a random place, then go to the end and loop around
        self.seed_seq = self.DATA[100000:100000 + 1000]
        self.sess = sess

        """
        Placeholders for minibatch input and output data
        """
        self.X = tf.placeholder(tf.int32, [None, self.SEQ_LENGTH]) # the sequences
        self.Y = tf.placeholder(tf.int32, [None]) 
        X_onehot = tf.one_hot(self.X, self.NUM_CLASSES)
        Y_onehot = tf.one_hot(self.Y, self.NUM_CLASSES)

        """
        BUILDING THE GRAPH
        BUILDING THE GRAPH
        BUILDING THE GRAPH
        """

        # RNN layer
        Cell = tf.contrib.rnn.BasicLSTMCell(self.NUM_CELL_UNITS, state_is_tuple=True)
        # Dropout = tf.contrib.rnn.DropoutWrapper(Cell, input_keep_prob=DROPOUT_KEEP_PROB)
        # Cells = tf.contrib.rnn.MultiRNNCell([Dropout] * self.NUM_LSTM_CELLS)
        Cells = tf.contrib.rnn.MultiRNNCell([Cell] * self.NUM_LSTM_CELLS)
        self.InitState = Cells.zero_state(tf.shape(self.X)[0], tf.float32)
        Output, self.State = tf.nn.dynamic_rnn(Cells, X_onehot, initial_state=self.InitState, dtype=tf.float32)

        # Fully connected layer mapping RNN output to classes
        W = tf.Variable(tf.truncated_normal([self.NUM_CELL_UNITS, self.NUM_CLASSES]), dtype=tf.float32)
        b = tf.Variable(tf.constant(0, shape=[self.NUM_CLASSES], dtype=tf.float32))
        Last = tf.gather(tf.transpose(Output, [1, 0, 2]), self.SEQ_LENGTH - 1)
        Logits = tf.matmul(Last, W) + b

        # Results
        Predictions = tf.nn.softmax(Logits)
        Corrects = tf.equal(tf.argmax(Y_onehot, axis=1), tf.argmax(Predictions, axis=1))
        self.Accuracy = tf.reduce_mean(tf.cast(Corrects, tf.float32))

        """
        Loss and optimization
        """
        Losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=Logits)
        self.Loss = tf.reduce_mean(Losses)
        self.Minimize = tf.train.AdamOptimizer().minimize(self.Loss)

        self.sess.run(tf.global_variables_initializer())

        batchX, _ = self.nextTrainBatch()
        self.state = self.sess.run(self.InitState, {self.X: batchX})

    def train_batch(self):
        self.batchX, self.batchY = self.nextTrainBatch()
        minimize, self.loss, self.state = self.sess.run([
            self.Minimize, self.Loss, self.State
        ], {
            self.X: self.batchX, self.Y: self.batchY, self.InitState: self.state
        })

    def sample(self):
        self.seed_seq = self.seed_seq[-self.SEQ_LENGTH:]
        output = []    

        print "SEED"
        print "----"
        print self.seed_seq
        print "----"
        
        GEN_STR_LEN = 300    
        for i in xrange(GEN_STR_LEN):
            testX = genTestXFromString(self.seed_seq)
            predictions = self.sess.run(Predictions, {X: testX})
            char = char_distr_to_char(predictions)
            output.append(char)
            self.seed_seq = (self.seed_seq + char)[-self.SEQ_LENGTH:]

        print "GENERATED"
        print "========="
        print "".join(output)
        print "========="

    def nextTrainBatch(self):
        """
        Generate training batches
        """
        batchX = []
        batchY = []
        for i in xrange(self.BATCH_SIZE):
            x = [p for p in self.DATA[self.batch_ptr + i:self.batch_ptr + self.SEQ_LENGTH + i]]
            y = self.DATA[self.batch_ptr + self.SEQ_LENGTH + i]
            batchX.append(x)
            batchY.append(y)
            self.batch_ptr = (self.batch_ptr + 1) % self.NUM_TRAIN
        batchX = np.array(batchX)
        batchY = np.array(batchY)
        return batchX, batchY

    def genTestXFromString(s):
        ixes = [[CHAR_TO_IX[ch] for ch in s]]
        return ixes

    def ixes_to_string(ixes):   
        """
        Convert a list of indices to a string
        """
        pass
        # return "".join([IX_TO_CHAR[i] for i in ixes])

    def char_distr_to_char_ix(predictions):
        """
        Prediction distributions for a single char, i.e. the probability of a
        character being correct is given by the corresponding probability in
        predictions. NOTE. The correct char is not the one with the highest
        probability: np.argmax(predictions) is not the character you're looking for.
        Instead we choose a char index out of range(self.NUM_CLASSES) based on
        probabilities given by predictions.
        """
        return np.random.choice(range(self.NUM_CLASSES), p=predictions.ravel())

    def char_distr_to_char(predictions):
        pass
        # return IX_TO_CHAR[char_distr_to_char_ix(predictions)]

    def get_accuracy(self):
        return self.sess.run(self.Accuracy, {self.X: self.batchX, self.Y: self.batchY})

    def get_loss(self):
        return self.loss