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
# print "_input_data", session.run(_input_data, feed_dict)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
_initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
# print "_initial_state", session.run(_initial_state, feed_dict)

# where did this guy come from?
embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
session.run(tf.global_variables_initializer())
# print "embedding", session.run(embedding, feed_dict)

# Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding, _input_data)  # shape=(30, 20, 200)
# print "inputs", inputs
# print "inputs shape", session.run(inputs, feed_dict).shape
# print "first input", session.run(inputs[0], feed_dict)

outputs, new_state = tf.nn.dynamic_rnn(
    stacked_lstm, inputs, initial_state=_initial_state)
session.run(tf.global_variables_initializer())
# print "first output", session.run(outputs[0], feed_dict)
# print "output", session.run(outputs, feed_dict)

output = tf.reshape(outputs, [-1, size])
# print "first output shape", session.run(output[0], feed_dict).shape

softmax_w = tf.get_variable("softmax_w", [size, vocab_size])  # [200x1000]
softmax_b = tf.get_variable("softmax_b", [vocab_size])  # [1x1000]
logits = tf.matmul(output, softmax_w) + softmax_b

session.run(tf.global_variables_initializer())
logi = session.run(logits, feed_dict)
# print "logit", logi
# print "logit shape", logi.shape

First_word_output_probablity = logi[0]
print "First_word_output_probablity", First_word_output_probablity

embedding_array = session.run(embedding, feed_dict)
print "First_word_output_probablity argmax", np.argmax(First_word_output_probablity)
targ = session.run(tf.reshape(_targets, [-1]), feed_dict)
first_word_target_code = targ[0]
print "first_word_target_code", first_word_target_code
first_word_target_vec = session.run(tf.nn.embedding_lookup(embedding, targ[0]))
print "first_word_target_vec", first_word_target_vec
print "first_word_target_vec.shape", first_word_target_vec.shape

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [logits], [tf.reshape(_targets, [-1])], [tf.ones([batch_size * num_steps])])
# session.run(loss, feed_dict)
cost = tf.reduce_sum(loss) / batch_size
session.run(tf.global_variables_initializer())
session.run(cost, feed_dict)

final_state = new_state

lr = tf.Variable(0.0, trainable=False)
optimizer = tf.train.GradientDescentOptimizer(lr)

# Get all TensorFlow variables marked as "trainable" (i.e. all of them
# except lr, which we just created)
tvars = tf.trainable_variables()
# print "trainable variables", session.run(tvars)

# Gradient example
var_x = tf.placeholder(tf.float32)
var_y = tf.placeholder(tf.float32)
func_test = 2.0 * var_x * var_x + 3.0 * var_x * var_y
session.run(tf.global_variables_initializer())
feed = {var_x: 1.0, var_y: 2.0}
print "function value", session.run(func_test, feed)

xgrad = tf.gradients(func_test, [var_x])
ygrad = tf.gradients(func_test, [var_y])
print "xgrad", session.run(xgrad, feed)
print "ygrad", session.run(ygrad, feed)

grad_t_list = tf.gradients(cost, tvars)
# print "grad_t_list", session.run(grad_t_list, feed_dict)

# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
# print "grads", session.run(grads, feed_dict)

train_op = optimizer.apply_gradients(zip(grads, tvars))
session.run(tf.global_variables_initializer())
session.run(train_op, feed_dict)


class PTBModel(object):

    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        size = hidden_size
        self.vocab_size = vocab_size

        #######################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        #######################################################################
        self._input_data = tf.placeholder(
            tf.int32, [batch_size, num_steps])  # [30#20]
        self._targets = tf.placeholder(
            tf.int32, [batch_size, num_steps])  # [30#20]

        #######################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        #######################################################################
        # Create the LSTM unit.
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument n_hidden(size=200) of BasicLSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # Size is the same as the size of our hidden layer, and no bias is added to the Forget Gate.
        # LSTM cell processes one word at a time and computes probabilities of
        # the possible continuations of the sentence.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout wrapper for our LSTM unit
        # This is an optimization of the LSTM output, but is not needed at all
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)

        # By taking in the LSTM cells as parameters, the MultiRNNCell function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of multiple simple cells.
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the
        # network is initialized with a vector of zeros and gets updated after
        # reading each word.
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            embedding = tf.get_variable(
                "embedding", [vocab_size, size])  # [10000x200]
            # Define where to get the data for our embeddings from
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout addition for our inputs
        # This is an optimization of the input processing and is not needed at
        # all
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.
        # In step 2,  second word of each of the b sentences is input in parallel.
        # The parallelism is only for efficiency.
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly.
        # All the computations involving the words of all sentences in a batch
        # at a given time step are done in parallel.

        #######################################################################
        # Instanciating our RNN model and retrieving the structure for returning the outputs and the state #
        #######################################################################

        outputs, state = tf.nn.dynamic_rnn(
            stacked_lstm, inputs, initial_state=self._initial_state)

        #######################################################################
        # Creating a logistic unit to return the probability of the output word #
        #######################################################################
        output = tf.reshape(outputs, [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size])  # [200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size])  # [1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b

        #######################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #######################################################################
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        # Store the final state
        self._final_state = state

        # Everything after this point is relevant only for training
        if not is_training:
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them
        # except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Helper functions for our LSTM RNN class

    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data

    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets

    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state

    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost

    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state

    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr

    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op

#########################################################################
# run_epoch takes as parameters the current session, the model instance, the
# data to be fed, and the operation to be run #
# #########################################################################


def run_epoch(session, m, data, eval_op, verbose=False):

    # Define the epoch size based on the length of the data, batch size and
    # the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    #m.initial_state = tf.convert_to_tensor(m.initial_state)
    #state = m.initial_state.eval()
    state = session.run(m.initial_state)

    # For each step and data point
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):

        # Evaluate and return cost, state by running cost, final_state and the
        # function passed as parameter
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})

        # Add returned cost to costs (which keeps track of the total costs for
        # this epoch)
        costs += cost

        # Add number of steps to iteration counter
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
                                                             iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is
    # evolving
    return np.exp(costs / iters)


# Reads the data and separates it into training data, validation data and
# testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

# Initializes the Execution Graph and the Session
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    # Instantiates the model for training
    # tf.variable_scope add a prefix to the variables created with
    # tf.get_variable
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True)

    # Reuses the trained parameters for the validation and testing models
    # They are different instances but use the same variables for weights and
    # biases, they just don't change when data is input
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False)
        mtest = PTBModel(is_training=False)

    # Initialize all variables
    tf.global_variables_initializer().run()

    for i in range(max_max_epoch):
        # Define the decay for this epoch
        lr_decay = decay ** max(i - max_epoch, 0.0)

        # Set the decayed learning rate as the learning rate for this epoch
        m.assign_lr(session, learning_rate * lr_decay)

        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))

        # Run the loop for this epoch in the training model
        train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                     verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))

        # Run the loop for this epoch in the validation model
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    # Run the loop in the testing model to see how effective was our training
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())

    print("Test Perplexity: %.3f" % test_perplexity)
