"""
Generate ABC training data by running:

    cd ~/ai/audio
    bash bin/gen_master_abc.sh

This will concatenate all the ABC files in data/abc/ into one master file (data/abc/master.abc).

Optionally preprocess master.abc with

    python bin\preprocess_abc.py data\abc\master.abc > data\abc\master_processed.abc

Then run this file to train and generate audio:

    python audio_abc.py data/abc/master.abc

"""

import sys
sys.path.append("../tensorflow/rnn/")
import time
import sys
import glob
import random
import string
import numpy as np
# np.set_printoptions(threshold=np.nan) # lets you print full numpy arrays
import tensorflow as tf
# import audio_utils
from rnn import RNN
from lib import utils

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
    "SEQ_LENGTH": 50,
    "NUM_CELL_UNITS": 256,
    "NUM_LSTM_CELLS": 3,
    "NUM_CLASSES": VOCAB_SIZE,
    "SAVE_DST": "./save_abc/",
})

output_file = open("output/output.abc", "w")
t = time.time()
i = 0
while True: 
    batchX, batchY = model.train_batch()

    if i % 1 == 0:
        print("TRAINING")
        print("========")
        print(utils.batchX_to_str(IX_TO_CHAR, batchX))
        print("========")

        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()

    if i % 10 == 0:
        accuracy = model.get_accuracy()
        loss = model.get_loss()
        print('Batch: {:2d}, elapsed: {:.4f}, loss: {:.4f}, accuracy: {:3.1f} %'.format(i, time.time() - t, loss, 100 * accuracy))
        t = time.time()

    """
    GENERATE AUDIO.

    Once the network has learned enough, generate a music sample every iteration and 
    see if it's simply repeating (with slight variation) what is currently being trained
    right now.

    Another test to try: once the network has learned enough, let
    the network generate on its own without training at the same time.
    """
    # if i % 1 == 0:
    if i % 50 == 0:    
    # if i % 50 == 0 and i != 0:    
        sample = model.sample(100)
        sample = utils.ixes_to_string(IX_TO_CHAR, sample)

        print("SAMPLE")
        print("======")
        print(sample)
        print("======")

        output_file.write(sample)
        output_file.write("\n")
        output_file.flush()

        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()        

    if i % 50 == 0 and i != 0:
        model.save(i)

    i += 1

sess.close()
