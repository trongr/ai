"""
Generate MIDI csv training data by running:

    cd ~/ai/audio
    bash bin/gen_master_csv.sh

This will convert all the MIDI files in midi/ into their text/csv versions, and
concatenate them into a single csv/master.csv file.

Then run this file to train and generate audio:

    python audio_midi.py csv/master.csv

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
import audio_utils
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
    "NUM_CELL_UNITS": 256,
    "NUM_LSTM_CELLS": 3,
    "NUM_CLASSES": VOCAB_SIZE,
    "SAVE_DST": "./save_midi/",
})

output_csv = open("csv/output.csv", "w")
t = time.time()
i = 0
while True: 
    batchX, batchY = model.train_batch()

    if i % 10 == 0:
        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()

    if i % 100 == 0:
        accuracy = model.get_accuracy()
        loss = model.get_loss()
        print('Batch: {:2d}, elapsed: {:.4f}, loss: {:.4f}, accuracy: {:3.1f} %'.format(i, time.time() - t, loss, 100 * accuracy))
        t = time.time()

    if i % 100 == 0:    
        sample = model.sample(300)
        sample = ixes_to_string(sample)

        print "TRAINING"
        print "========"
        print ixes_to_string(batchX[0])
        print "========"

        print "SAMPLE"
        print "======"
        print sample 
        print "======"

        output_csv.write(sample)
        output_csv.write("\n")
        output_csv.flush()

        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()        

    if i % 500 == 0:
        model.save("./save_midi/audio", i)

    i += 1

sess.close()
