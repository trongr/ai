import numpy as np
import scipy.io.wavfile as wavfile

def read_wav(filename):
    rate, data = wavfile.read(filename)
    return rate, np.array(data, dtype=float)

def write_wav(filename, rate, data):
    # bits-per-sample and PCM/float will be determined by the data-type:
    # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.io.wavfile.write.html
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(filename, rate, scaled)