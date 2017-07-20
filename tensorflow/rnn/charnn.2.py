"""
Prepare input audio files with:
ffmpeg -i pingpong.wav -vn -ac 1 -c:a pcm_u8 -ar 8000 pingpong8bit8khz.wav
which converts a wav file to one channel, 8 bit, 8KHz wav.
"""

import sys
import time
import sys
import glob
import random
import string
import numpy as np
# np.set_printoptions(threshold=np.nan) # lets you print full numpy arrays
import tensorflow as tf

from rnn import RNN

def ixes_to_string(ixes):   
    """
    Convert a list of indices to a string
    """
    return "".join([IX_TO_CHAR[i] for i in ixes])

FILENAME = sys.argv[1]
DATA = open(FILENAME, 'r').read()
# DATA = DATA[100000:110000]
CHARS = list(set(DATA))
DATA_SIZE, VOCAB_SIZE = len(DATA), len(CHARS)
CHAR_TO_IX = {ch: i for i, ch in enumerate(CHARS)}
IX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}
data = [CHAR_TO_IX[ch] for ch in DATA]

sess = tf.Session()
model = RNN(sess, data, {
    "SEQ_LENGTH": 100,
    "NUM_CELL_UNITS": 64,
    "NUM_LSTM_CELLS": 3,
    "NUM_CLASSES": VOCAB_SIZE,
})

Saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1) 
save_files = glob.glob('./save/*')
if save_files:
    Saver.restore(sess, tf.train.latest_checkpoint('./save/'))

t = time.time()
i = 0
while True: 
    model.train_batch()

    if i % 10 == 0:
        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()

    if i % 100 == 0:
        accuracy = model.get_accuracy()
        loss = model.get_loss()
        print('Batch: {:2d}, elapsed: {:.4f}, loss: {:.4f}, accuracy: {:3.1f} %'
            .format(i, time.time() - t, loss, 100 * accuracy))

    if i % 100 == 0:    
        sample = model.sample(300)
        sample = ixes_to_string(sample)

        print "SAMPLE"
        print "======"
        print sample 
        print "======"

        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()        

    if i % 1000 == 0:
        Saver.save(sess, "./save/char", global_step=i)

    i += 1

sess.close()
