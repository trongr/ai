import numpy as np

"""
sliding x window over h:

    6 = 2 * 3
    11 = 1 * 3 + 2 * 4
    14 = 0 * 3 + 1 * 4 + 2 * 5
    5 = 0 * 4 + 1 * 5
    0 = 0 * 5
"""
h = [2, 1, 0]
x = [3, 4, 5]
y = np.convolve(x, h)

print "h", h
print "x", x
print "y", y  # [6, 11, 14, 5, 0]
print

h = [1, 2, 5, 4]
x = [6, 2]
y = np.convolve(x, h)

print "h", h
print "x", x
print "y", y  # [ 6 14 34 34  8]
print

h = [1, 2, 5, 4]
x = [6, 2]
y = np.convolve(x, h, "full")  # default is full

print "full"
print "h", h
print "x", x
print "y", y  # [ 6 14 34 34  8]
print

h = [1, 2, 5, 4]
x = [6, 2]
y = np.convolve(x, h, "valid")  # default is valid

print "valid"
print "h", h
print "x", x
print "y", y  # [ 6 14 34 34  8]
print

h = [1, 2, 5, 4]
x = [6, 2]
y = np.convolve(x, h, "same")  # default is same

print "same"
print "h", h
print "x", x
print "y", y  # [ 6 14 34 34  8]
print
