import numpy as np
from scipy import signal as sg

I = [[255,   7,  3],
     [212, 240,  4],
     [218, 216, 230]]
g = [[-1, 1]]

print "I"
print np.array(I)
print "g"
print np.array(g)
print

print ('Without zero padding (valid)\n')
print ('{0} \n'.format(sg.convolve(I, g, 'valid')))

print ('With zero padding (default full)\n')
print sg.convolve(I, g)
print

print ('FULL\n')
print sg.convolve(I, g, "full")

print ('same\n')
print sg.convolve(I, g, "same")
