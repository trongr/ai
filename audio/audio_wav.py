"""
Convert a wav file to one channel, 8 bit, 8KHz wav:
ffmpeg -i pingpong.wav -vn -ac 1 -c:a pcm_u8 -ar 8000 pingpong8bit8khz.wav
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

# TODO. Change NUM_CLASSES below if changing audio bit depth
WAV_IN = "wav/pingpong.wav"
WAV_IN = "wav/pingpong8bit8khz.wav"
rate, data = audio_utils.read_wav(WAV_IN)

sess = tf.Session()
model = RNN(sess, data, {
    "SEQ_LENGTH": 100,
    "NUM_CELL_UNITS": 64,
    "NUM_LSTM_CELLS": 3,
    "NUM_CLASSES": 256, # 8 bit audio
    "SAVE_DST": "./save/",
})

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
        print('Batch: {:2d}, elapsed: {:.4f}, loss: {:.4f}, accuracy: {:3.1f} %'.format(i, time.time() - t, loss, 100 * accuracy))
        t = time.time()

    # if i != 0 and i % 1000 == 0:
    if i % 10000 == 0:    
        bitrate = 8000
        duration_in_seconds = 30
        show_every = 50
        sample = model.sample(bitrate * duration_in_seconds, show_every) 
        print('Batch: {:2d}, elapsed: {:.4f}'.format(i, time.time() - t))
        t = time.time()                
        
        WAV_OUT = "samples/pingpong_" + str(i) + ".wav"
        audio_utils.write_wav(WAV_OUT, rate, sample)
        print "Generating", WAV_OUT

    if i % 1000 == 0:
        model.save("./save/audio", i)

    i += 1

sess.close()
