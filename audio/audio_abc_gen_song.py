# coding: utf-8
"""
Generate song continuing from input song:

    cd ~/ai/audio
    python audio_abc_gen_song.py input_song.abc

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

# TODO. Replace this list with new char set if DATA contains new chars
CHARS = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\x80', '\x89', '\x93', '\x94', '\x96', '\x98', '\x99', '\xa5', '\xa9', '\xb8', '\xc0', '\xc2\xa5', '\xc2\xa9', '\xc2\xb8', '\xc3', '\xc3\x80', '\xc3\x83', '\xc3\x8a', '\xc3\xa0', '\xc3\xa2', '\xc3\xa9', '\xc3\xb2', '\xc3\xb3', '\xc3\xb9', '\xc3\xbf', '\xca', '\xe0', '\xe2', '\xe2\x80\x93', '\xe2\x80\x9c', '\xe2\x82\xac', '\xe9', '\xff']
CHARS.extend(list(set(DATA)))
CHARS = list(set(CHARS))
CHARS.sort()
print("Char set", CHARS)

DATA_SIZE, VOCAB_SIZE = len(DATA), len(CHARS)
CHAR_TO_IX = {ch: i for i, ch in enumerate(CHARS)}
IX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}
data = [CHAR_TO_IX[ch] for ch in DATA]

sess = tf.Session()

model = RNN(sess, data, {
    "SEQ_LENGTH": 50,
    "NUM_CELL_UNITS": 256,
    "NUM_LSTM_CELLS": 3, # This is ignored. Hardcoding num cells in RNN lib for now.
    "NUM_CLASSES": VOCAB_SIZE,
    "SAVE_DST": "./save_abc/",
    "SEED": data
})

sample = model.sample(1000)
sample = utils.ixes_to_string(IX_TO_CHAR, sample)

print("SAMPLE")
print("======")
print(sample)
print("======")

# output_file = open("output/output" + str(i) + ".abc", "w", encoding="utf-8")            
output_file = open("output/tmp/output_song_contd.abc", "w")
try: 
    output_file.write(sample)
    output_file.flush()
finally:
    output_file.close()

sess.close()
