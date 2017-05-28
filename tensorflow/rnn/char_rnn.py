import sys
import numpy as np

filename = sys.argv[1]
data = open(filename, 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# print "chars", chars
# print "char_to_ix", char_to_ix
# print "ix_to_char", ix_to_char
# print "data", data

hidden_size = 100  # size of hidden layer of neurons
batch_size = 100  # mini batch size
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1


def genBatch():
    """
    Return a random batch of training text and target chars
    """
    X = []
    Y = []
    for i in xrange(batch_size):
        x, y = genExample()
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)[:, None]


def genExample():
    """
    Return a random line of text and target char
    """
    # in case data_size == seq_length and we run into infinite recursion with
    # the call to return genExample() below
    if data_size <= seq_length:
        idx = 0
    else:
        idx = np.random.randint(0, data_size)

    # near the end of the text file: training text has no target char following
    # it... [NOTE. This recursion has to return at some point because data_size
    # > seq_length]
    if idx + seq_length >= data_size:
        return genExample()  # ...so get another example

    x = [char_to_ix[ch] for ch in data[idx:idx + seq_length]]
    # targets = [char_to_ix[ch] for ch in data[idx + 1:idx + seq_length + 1]]
    # print "targets", targets
    y = char_to_ix[data[idx + seq_length]]

    return x, y


print "Example", genExample()
print "Batch", genBatch()

# TODO

# while True:
#     inputs, targets = genBatch()

#     # sample from the model now and then
#     if n % 100 == 0:
#         sample_ix = sample(hprev, inputs[0], 200)
#         txt = ''.join(ix_to_char[ix] for ix in sample_ix)
#         print '----\n %s \n----' % (txt, )
