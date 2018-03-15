from __future__ import print_function, division
import os
import time
import math
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import utils

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
x_dim = 784 # 28 * 28, dimension of each image
noise_dim = 64

mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)

def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_images(dir, images, it):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = utils.deprocess_img(img)
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))

    imgpath = dir + "/" + str(it).zfill(10) + ".jpg"
    print("Saving img " + imgpath)
    fig.savefig(imgpath)
    plt.close(fig)

def preprocess_img(x):
    return 2 * x - 1.0

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def discriminator(x):
    # x ~ (N, x_dim) = 28 * 28 = 784
    with tf.variable_scope("discriminator"):
        fc0 = tf.layers.dense(inputs=x, units=16 * 16, activation=leaky_relu)
        rs0 = tf.reshape(fc0, [-1, 16, 16, 1])

        c1 = tf.layers.conv2d(inputs=rs0, filters=32, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2) # (-1, 14, 14, 32)

        c2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2) # (-1, 7, 7, 32)

        rs1 = tf.reshape(p2, [-1, 7 * 7 * 32])
        fc1 = tf.layers.dense(inputs=rs1, units=1024, activation=leaky_relu)
        logits = tf.layers.dense(inputs=fc1, units=1)

        return logits

def generator(z):
    # z ~ (N, noise_dim) = 64
    with tf.variable_scope("generator"):
        rs0 = tf.reshape(z, [-1, 8, 8, 1])

        c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2) # (-1, 4, 4, 16)

        c2 = tf.layers.conv2d(inputs=p1, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2) # (-1, 2, 2, 16)

        rs2 = tf.reshape(p2, [-1, 2 * 2 * 16])
        fc1 = tf.layers.dense(inputs=rs2, units=256, activation=leaky_relu)
        img = tf.layers.dense(inputs=fc1, units=x_dim, activation=tf.tanh)
        return img

def log(x):
    return tf.log(x + 1e-8)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, x_dim])
z = tf.placeholder(tf.float32, shape=[None, noise_dim])

with tf.variable_scope("") as scope:
    G_sample = generator(z)
    D_real = discriminator(utils.preprocess_img(x)) # scale images to be -1 to 1
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    D_fake = discriminator(G_sample)

D_target = 1. / batch_size
G_target = 1. / batch_size
Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))
D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
G_loss = tf.reduce_sum(G_target * D_fake) + log(Z)

dlr, glr, beta1 = 1e-3, 1e-3, 0.5
D_solver = tf.train.AdamOptimizer(learning_rate=dlr, beta1=beta1)
G_solver = tf.train.AdamOptimizer(learning_rate=glr, beta1=beta1)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
with tf.control_dependencies(D_extra_step):
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
with tf.control_dependencies(G_extra_step):
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

def train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,
    save_img_every=250, print_every=50, batch_size=128, num_epoch=10):
    out_dir = "out"
    save_dir = "save"
    mkdir_p(out_dir)
    mkdir_p(save_dir)

    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
    t = time.time()
    for it in range(max_iter):
        xmb, _ = mnist.train.next_batch(batch_size)
        z_noise = sample_z(batch_size, noise_dim)

        if it % save_img_every == 0:
            samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise})
            save_images(out_dir, samples[:49], it)

        # train G twice for every D train step. see if that helps learning.
        _, D_loss_curr, _ = sess.run([D_train_step, D_loss, D_extra_step], feed_dict={x: xmb, z: z_noise})
        _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise})
        _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise})

        if math.isnan(D_loss_curr) or math.isnan(G_loss_curr):
            print("Loss is nan: D: {:.4}, G: {:.4}".format(D_loss_curr, G_loss_curr))
            exit()
        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}, Elapsed: {:.4}'.format(it, D_loss_curr, G_loss_curr, time.time() - t))
            t = time.time()

        if it % 10 == 0:
            Saver.save(sess, save_dir + "/conv-g-conv-d-softmax-mnist", global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, save_img_every=25, print_every=1, num_epoch=1000)
