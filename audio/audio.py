import numpy as np
# np.set_printoptions(threshold=np.nan) # lets you print full numpy arrays
import audio_utils

WAV_IN = "pingpong.wav"
WAV_IN = "pingpong8bit8khz.wav"
rate, data = audio_utils.read_wav(WAV_IN)

WAV_OUT = "pingpongout.wav"
audio_utils.write_wav(WAV_OUT, rate, data)