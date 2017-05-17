import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

im = misc.imread("./lena.png").astype(np.float)
grayim = np.dot(im[..., :3], [0.299, 0.587, 0.114])

# plt.imshow(im)
# plt.xlabel(" Float Image ")
# plt.show()

# plt.imshow(grayim, cmap=plt.get_cmap("gray"))
# plt.xlabel(" Gray Scale Image ")
# plt.show()

print "Original image shape", grayim.shape

Image = np.expand_dims(np.expand_dims(grayim, 0), -1)
print "Expanded image shape", Image.shape

img = tf.placeholder(tf.float32, [None, 512, 512, 1])
print "Image placeholder", img.get_shape().as_list()

weights = tf.Variable(tf.truncated_normal([5, 5, 1, 1], stddev=0.05))

ConOut1 = tf.nn.conv2d(input=img, filter=weights, strides=[
                       1, 1, 1, 1], padding='SAME')
ConOut2 = tf.nn.conv2d(input=img, filter=weights, strides=[
                       1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result1 = sess.run(ConOut1, feed_dict={img: Image})
    result2 = sess.run(ConOut2, feed_dict={img: Image})
    print "result 1.shape", result1.shape
    print "result 2.shape", result2.shape

    vec = np.reshape(result1, (1, -1))
    image = np.reshape(vec, (512, 512))
    print image.shape

    vec2 = np.reshape(result2, (1, -1))
    image2 = np.reshape(vec2, (508, 508))
    print image2.shape

    # plt.imshow(image, cmap=plt.get_cmap("gray"))
    # plt.xlabel(" SAME Padding ")
    # plt.show()

    # plt.imshow(image2, cmap=plt.get_cmap("gray"))
    # plt.xlabel(" VALID Padding ")
    # plt.show()
