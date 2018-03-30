from __future__ import print_function, division
import os
import time
import math
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import misc
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)
import utils

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

batch_size = 64
img_h = 218
img_w = 178
img_c = 3
w_to_h = 1.0 * img_w / img_h
x_dim = 116412  # 218 * 178 * 3 dimension of each image
noise_dim = 64
D_input_dim = noise_dim + x_dim


def load_images(img_dir):
    img_paths = []
    for img in os.listdir(img_dir):
        img_paths.append(os.path.join(img_dir, img))
    total = len(img_paths)
    i = 0
    while (True):
        if i + batch_size >= total:
            i = 0
            continue
        images = []
        for j in range(batch_size):
            images.append(misc.imread(img_paths[i + j]))
        images = np.reshape(np.asarray(images), [-1, x_dim])
        images = utils.preprocess_img_rgb(images)
        assert not np.any(np.isnan(images)), "Images should not contain nan's"
        yield(images)
        i = (i + batch_size) % total


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_images(dir, images, it):
    fig = plt.figure(figsize=(8 * w_to_h, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        img = utils.deprocess_img(img)
        plt.imshow(img.reshape([img_h, img_w, img_c]))

    imgpath = dir + "/" + str(it).zfill(10) + ".jpg"
    print("Saving img " + imgpath)
    fig.savefig(imgpath)
    plt.close(fig)


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
    # x ~ (N, x_dim)
    with tf.variable_scope("discriminator"):
        # Cluster 1
        fc0 = tf.layers.dense(inputs=x, units=16 * 16, activation=leaky_relu)
        rs0 = tf.reshape(fc0, [-1, 16, 16, 1])

        c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2)  # (-1, 8, 8, 16)

        c2 = tf.layers.conv2d(inputs=p1, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2)  # (-1, 4, 4, 16)

        rs3 = tf.reshape(p2, [-1, 4 * 4 * 16])

        # Cluster 2
        fc4 = tf.layers.dense(inputs=rs3, units=16 * 16, activation=leaky_relu)
        rs4 = tf.reshape(fc4, [-1, 16, 16, 1])

        c5 = tf.layers.conv2d(inputs=rs4, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p5 = tf.layers.max_pooling2d(inputs=c5, pool_size=2, strides=2)  # (-1, 8, 8, 16)

        c6 = tf.layers.conv2d(inputs=p5, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p6 = tf.layers.max_pooling2d(inputs=c6, pool_size=2, strides=2)  # (-1, 4, 4, 16)

        rs7 = tf.reshape(p6, [-1, 4 * 4 * 16])

        # Tail cluster 3
        fc7 = tf.layers.dense(inputs=rs7, units=16 * 16, activation=leaky_relu)
        logits = tf.layers.dense(inputs=fc7, units=1)

        return logits


def generator(z, keep_prob):
    # z ~ (N, D_input_dim). D_input_dim = 64 + 218 * 178 * 3 = 116476 = 4 * 37 * 787
    with tf.variable_scope("generator"):
        # Cluster 1
        fc0 = tf.layers.dense(inputs=z, units=8 * 8, activation=leaky_relu)  # reduce input dimension otw it'll blow up MEM
        rs0 = tf.reshape(fc0, [-1, 8, 8, 1])

        c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2)  # (-1, 4, 4, 16)

        c2 = tf.layers.conv2d(inputs=p1, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2)  # (-1, 2, 2, 16)

        rs3 = tf.reshape(p2, [-1, 2 * 2 * 16])

        # Cluster 2
        fc4 = tf.layers.dense(inputs=rs3, units=8 * 8, activation=leaky_relu)
        rs4 = tf.reshape(fc4, [-1, 8, 8, 1])

        c5 = tf.layers.conv2d(inputs=rs4, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p5 = tf.layers.max_pooling2d(inputs=c5, pool_size=2, strides=2)  # (-1, 4, 4, 16)

        c6 = tf.layers.conv2d(inputs=p5, filters=16, kernel_size=5, strides=1, padding='same', activation=leaky_relu)
        p6 = tf.layers.max_pooling2d(inputs=c6, pool_size=2, strides=2)  # (-1, 2, 2, 16)

        rs7 = tf.reshape(p6, [-1, 2 * 2 * 16])

        # Tail cluster 3
        fc7 = tf.layers.dense(inputs=rs7, units=16 * 16, activation=leaky_relu)
        img = tf.layers.dense(inputs=fc7, units=x_dim, activation=tf.tanh)
        return img


def log(x):
    return tf.log(x + 1e-8)


def gan_loss(logits_real, logits_fake):
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_fake),
            logits=logits_fake))

    D_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_real),
            logits=logits_real)) \
        + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(logits_fake),
            logits=logits_fake))

    return D_loss, G_loss


tf.reset_default_graph()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x-input")
    z = tf.placeholder(tf.float32, shape=[None, D_input_dim], name="z-input")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope("") as scope:
    G_sample = generator(z, keep_prob)
    D_real = discriminator(x)
    scope.reuse_variables()  # Re-use discriminator weights on new inputs
    D_fake = discriminator(G_sample)

# D_loss, G_loss = gan_loss(D_real, D_fake)

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

img_dir = "./data/img_align_celeba/"
out_dir = "out"
prefix = "conv-g-conv-d-softmax-celeba-deep-00"
save_dir = "save"
save_dir_prefix = save_dir + "/" + prefix
logs_path = "logs/" + prefix
mkdir_p(out_dir)
mkdir_p(save_dir)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


def train(sess, G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=250, print_every=50, max_iter=1000000):
    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    samples = None
    batches = load_images(img_dir)
    t = time.time()
    for it in range(max_iter):
        # xmb = batches.next()
        xmb = next(batches)
        if samples is None:  # Create all noise in the beginning
            z_noise = sample_z(batch_size, D_input_dim)
        else:  # Every iteration afterwards add samples to noise
            z_noise = sample_z(batch_size, noise_dim)
            z_noise = np.hstack((z_noise, samples))

        # train G twice for every D train step. see if that helps learning.
        _, D_loss_curr, _, summary = sess.run([D_train_step, D_loss, D_extra_step, summary_op], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise, keep_prob: 1.0})

        if it % save_img_every == 0:
            save_images(out_dir, samples[:64], it)

        if math.isnan(D_loss_curr) or math.isnan(G_loss_curr):
            print("D or G loss is nan", D_loss_curr, G_loss_curr)
            exit()

        if it % print_every == 0:  # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}, Elapsed: {:.4}'.format(it, D_loss_curr, G_loss_curr, time.time() - t))
            t = time.time()

        if it % 10 == 0:
            writer.add_summary(summary, global_step=it)
            Saver.save(sess, save_dir_prefix, global_step=it)


with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    train(sess, G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=25, print_every=1, max_iter=1000000)
