import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("bird.jpg")

# uses the ITU-R 601-2 Luma transform (there are several
# ways to convert an image to grey scale)
image_gr = im.convert("L")
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)
print "image shape", arr.shape
print "image min, max", np.min(arr), np.max(arr)

# imgplot = plt.imshow(arr)
# # you can experiment different colormaps (ocean, spring, summer, winter,
# # autumn)
# imgplot.set_cmap('autumn')
# print("\n Input image converted to gray scale: \n")
# plt.show(imgplot)

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
])

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
print "grad.shape", grad.shape
print "grad", grad

print('GRADIENT MAGNITUDE - Feature map')

# fig, aux = plt.subplots(figsize=(10, 10))
# aux.imshow(np.absolute(grad), cmap='gray')
# # aux.imshow(np.absolute(grad))
# plt.show(fig)

grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print "grad_biases > 255", grad_biases > 255
print "grad_biases[grad_biases > 255]", grad_biases[grad_biases > 255]
imgplot = plt.imshow(grad_biases)
plt.show(imgplot)

imgplot = plt.imshow(np.absolute(grad_biases), cmap='gray')
plt.show(imgplot)
