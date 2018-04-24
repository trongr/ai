from __future__ import print_function, division
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
from scipy import misc
import utils
import LossLib
sys.path.append("../utils/")
import MathLib
import TensorFlowLib

tf.app.flags.DEFINE_integer("train_size", None, "How many images to train on. Omit to train on all images.")
FLAGS = tf.app.flags.FLAGS

batch_size = 100
img_h = 218
img_w = 178
img_c = 3
x_dim = 116412  # 218, 178, 3 dimension of each image
noise_dim = 64  # NOTE. Generator hardcodes this so change that if this changes
img_dir = "./data/img_align_celeba/"
out_dir = "out"
prefix = os.path.basename(__file__)
save_dir = "save"
save_dir_prefix = save_dir + "/" + prefix
logs_path = "logs/" + prefix


def discriminator(x):
    with tf.variable_scope("discriminator"):
        # Cluster 1
        fc0 = tf.layers.dense(inputs=x, units=16 * 16, activation=MathLib.leaky_relu)
        rs0 = tf.reshape(fc0, [-1, 16, 16, 1])
        c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
        c2 = tf.layers.conv2d(inputs=c1, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
        rs3 = tf.reshape(c2, [-1, 16 * 16 * 16])
        # Cluster 2
        fc4 = tf.layers.dense(inputs=rs3, units=16 * 16, activation=MathLib.leaky_relu)
        rs4 = tf.reshape(fc4, [-1, 16, 16, 1])
        c5 = tf.layers.conv2d(inputs=rs4, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
        c6 = tf.layers.conv2d(inputs=c5, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
        rs7 = tf.reshape(c6, [-1, 16 * 16 * 16])
        # Tail cluster 3
        fc7 = tf.layers.dense(inputs=rs7, units=16 * 16, activation=MathLib.leaky_relu)
        logits = tf.layers.dense(inputs=fc7, units=1)
        return logits


def generator(z, keep_prob):  # z ~ (N, noise_dim) = 64
    with tf.variable_scope("generator"):
        # Cluster 1
        rs0 = tf.reshape(z, [-1, 8, 8, 1])
        c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu)
        c2 = tf.layers.conv2d(inputs=c1, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu)
        rs3 = tf.reshape(c2, [-1, 8 * 8 * 16])
        # Cluster 2
        fc4 = tf.layers.dense(inputs=rs3, units=8 * 8, activation=MathLib.leaky_relu)
        rs4 = tf.reshape(fc4, [-1, 8, 8, 1])
        c5 = tf.layers.conv2d(inputs=rs4, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu)
        c6 = tf.layers.conv2d(inputs=c5, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu)
        rs7 = tf.reshape(c6, [-1, 8 * 8 * 16])
        # Tail cluster 3
        fc7 = tf.layers.dense(inputs=rs7, units=16 * 16, activation=MathLib.leaky_relu)
        img = tf.layers.dense(inputs=fc7, units=x_dim, activation=tf.tanh)
        return img


tf.reset_default_graph()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x-input")
    z = tf.placeholder(tf.float32, shape=[None, noise_dim], name="z-input")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope("") as scope:
    G_sample = generator(z, keep_prob)
    D_real = discriminator(x)
    scope.reuse_variables()
    D_fake = discriminator(G_sample)

D_loss, G_loss = LossLib.VanillaGANLoss(D_real, D_fake)
D_train_step, G_train_step, D_extra_step, G_extra_step = TensorFlowLib.getTrainSteps(D_loss, G_loss)

utils.mkdir_p(out_dir)
utils.mkdir_p(save_dir)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


def train(sess, G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=250, print_every=50, max_iter=1000000):
    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    batches = utils.load_images(batch_size, x_dim, img_dir, total=FLAGS.train_size)
    t = time.time()
    for it in range(max_iter):
        xmb = next(batches)
        z_noise = MathLib.sample_z(batch_size, noise_dim)
        _, D_loss_curr, summary, _ = sess.run([D_train_step, D_loss, summary_op, D_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        if it % save_img_every == 0:
            samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise, keep_prob: 1.0})
            utils.save_images(out_dir, samples[:100], img_w, img_h, img_c, it)
        if math.isnan(D_loss_curr) or math.isnan(G_loss_curr):
            print("D or G loss is nan", D_loss_curr, G_loss_curr)
            exit()
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G: {:.4}, Elapsed: {:.4}'.format(it, D_loss_curr, G_loss_curr, time.time() - t))
            t = time.time()
        if it % 10 == 0:
            writer.add_summary(summary, global_step=it)
        if it % 100 == 0:
            Saver.save(sess, save_dir_prefix, global_step=it)


with TensorFlowLib.get_session() as sess:
    sess.run(tf.global_variables_initializer())
    train(sess, G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=25, print_every=1, max_iter=1000000)
