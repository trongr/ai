"""
USAGE.
======
python DeconvGConvDVanillaGAN.py --train False --noise_input=0.381935094072,0.993223977256,-0.917976787045,-0.738318301788,0.983937322845,0.568823153852,0.799077259729,0.308355393499,-0.273507900704,0.99261561491,-0.301363411266,-0.378580213994,0.553637597003,0.655594412744,-0.10589809701,-0.445392174697,0.127209381136,0.279231318717,-0.767187890619,0.517912382384,-0.982118335871,0.68021891118,0.550204859767,-0.405170726768,-0.209793820516,0.32421283446,0.655606459433,0.455121130887,0.444072844148,-0.723755365999,-0.876505903235,0.154755187644,0.103318084893,-0.813127115237,0.882995313363,-0.195894568946,-0.761815096228,0.991532449875,0.0581586407051,0.240098388243,0.905119550972,-0.593938262809,-0.0490899453885,-0.505825671087,-0.86150670744,-0.969691452214,0.265612969146,-0.67898421121,-0.849759991117,0.396833010409,-0.936391424904,-0.573455737039,-0.667525119719,-0.278111298132,-0.155129912759,0.979012054522,-0.31859680795,0.542003448302,-0.984675780767,0.223453406233,-0.825112411321,0.735301118248,-0.587375611399,-0.100033493553 --output output-00000001
Alternatively, import and use either train() or test().
"""

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

tf.app.flags.DEFINE_integer("train_size", None, "How many images to train on [None]. Omit to train on all images.")
tf.app.flags.DEFINE_boolean("train", False, "True for training, False for testing [False].")
tf.app.flags.DEFINE_string("noise_input", None, "List of random inputs [None]. Omit to create random image.")
tf.app.flags.DEFINE_string("output", "output", "Name of the output image [output].")
tf.app.flags.DEFINE_boolean("find_encoding", False, "Whether to find encoding of a ground-truth image [False].")
FLAGS = tf.app.flags.FLAGS

noise_input = None
if FLAGS.noise_input is not None:
    noise_input = [float(x) for x in FLAGS.noise_input.split(",")]

batch_size = 100
img_h = 218
img_w = 178
img_c = 3
x_dim = 116412  # 218, 178, 3 dimension of each image
noise_dim = 64
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


def encodingDiscriminator(x):
    """Discriminator used to find the encoding of a real image in backprop on input z."""
    fc0 = tf.layers.dense(inputs=x, units=16 * 16, activation=MathLib.leaky_relu)
    rs0 = tf.reshape(fc0, [-1, 16, 16, 1])
    c1 = tf.layers.conv2d(inputs=rs0, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
    c2 = tf.layers.conv2d(inputs=c1, filters=16, kernel_size=5, strides=1, padding='same', activation=MathLib.leaky_relu, use_bias=True)
    rs3 = tf.reshape(c2, [-1, 16 * 16 * 16])
    fc7 = tf.layers.dense(inputs=rs3, units=16 * 16, activation=MathLib.leaky_relu)
    logits = tf.layers.dense(inputs=fc7, units=1)
    return logits


def generator(z, keep_prob, training=False):
    with tf.variable_scope("generator"):
        fc0 = tf.layers.dense(inputs=z, units=1024, activation=MathLib.leaky_relu)
        bn0 = tf.layers.batch_normalization(fc0, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)

        fc1 = tf.layers.dense(inputs=bn0, units=7 * 7 * 128, activation=MathLib.leaky_relu)
        bn1 = tf.layers.batch_normalization(fc1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
        rs1 = tf.reshape(bn1, [-1, 7, 7, 128])

        ct2 = tf.layers.conv2d_transpose(rs1, filters=64, kernel_size=5, strides=2, padding='SAME', data_format='channels_last', activation=MathLib.leaky_relu, use_bias=True)  # (N, 14, 14, 64)
        bn2 = tf.layers.batch_normalization(ct2, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)  # Same as previous
        ct3 = tf.layers.conv2d_transpose(bn2, filters=1, kernel_size=5, strides=2, padding='SAME', data_format='channels_last', activation=MathLib.leaky_relu, use_bias=True)  # (N, 28, 28, 1)

        rs4 = tf.reshape(ct3, [-1, 28 * 28])
        img = tf.layers.dense(inputs=rs4, units=x_dim, activation=tf.tanh)
        return img


tf.reset_default_graph()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x-input")
    z = tf.placeholder(tf.float32, shape=[None, noise_dim], name="z-input")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    training = tf.placeholder(tf.bool, name="training")

with tf.variable_scope("") as scope:
    G_sample = generator(z, keep_prob, training)
    D_real = discriminator(x)
    scope.reuse_variables()
    D_fake = discriminator(G_sample)

D_loss, G_loss = LossLib.VanillaGANLoss(D_real, D_fake)

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

utils.mkdir_p(out_dir)
utils.mkdir_p(save_dir)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


def train(G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=250, print_every=50, max_iter=1000000):
    with TensorFlowLib.get_session() as sess:
        sess.run(tf.global_variables_initializer())
        Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        if glob.glob(save_dir + "/*"):
            Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        batches = utils.load_images(batch_size, x_dim, img_dir, total=FLAGS.train_size)
        t = time.time()
        for it in range(max_iter):
            xmb = next(batches)
            z_noise = MathLib.sample_z(batch_size, noise_dim)
            _, D_loss_curr, summary, _ = sess.run([D_train_step, D_loss, summary_op, D_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3, training: True})
            _, G_loss_curr, _ = sess.run([G_train_step, G_loss, G_extra_step], feed_dict={x: xmb, z: z_noise, keep_prob: 0.3, training: True})
            if it % save_img_every == 0:
                samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise, keep_prob: 0.3, training: True})  # TODO. Should this be False?
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


def test(noise_input, output):
    with TensorFlowLib.get_session() as sess:
        sess.run(tf.global_variables_initializer())
        Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        if glob.glob(save_dir + "/*"):
            Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        if noise_input is None:
            batch_size = 1
            z_noise = MathLib.sample_z(batch_size, noise_dim)
        else:
            z_noise = np.array(noise_input)
        samples = sess.run(G_sample, feed_dict={z: z_noise, keep_prob: 1.0, training: False})
        utils.save_images(out_dir, samples, img_w, img_h, img_c, output)
        print("z_noise:", ",".join(map(str, z_noise[0])))


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def backpropOnInputFromImage():
    """Find an approximate encoding z of the image by gradient descent on a random input z."""
    path = "./data/faces/evanrachelwood.jpg"
    image = utils.loadImage(path, x_dim)

    D_real = encodingDiscriminator(x)
    D_fake = encodingDiscriminator(G_sample)
    D_loss, _ = LossLib.VanillaGANLoss(D_real, D_fake)
    D_train_step = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(D_loss)

    Loss = tf.reduce_mean((image - G_sample) ** 2)
    zGrad = tf.gradients(Loss, z)[0]

    with TensorFlowLib.get_session() as sess:
        sess.run(tf.global_variables_initializer())

        optimistic_restore(sess, tf.train.latest_checkpoint(save_dir))
        # Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        # if glob.glob(save_dir + "/*"):
        #     Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        alpha = 0.1
        batch_size = 1
        max_iter = 1000
        z_noise = MathLib.sample_z(batch_size, noise_dim)
        for it in range(max_iter):
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: image, z: z_noise, keep_prob: 0.3, training: True})
            LossValue, zGradValue = sess.run([Loss, zGrad], feed_dict={z: z_noise, keep_prob: 1.0, training: False})
            z_noise -= alpha * zGradValue
            print('Iter: {}, Loss: {:.4}, D: {:.4}'.format(it, LossValue, D_loss_curr))
        print("Encoding:", ",".join(map(str, z_noise[0])))
        print("z shape", z_noise.shape)


def main():
    if FLAGS.find_encoding is True:
        backpropOnInputFromImage()
    elif FLAGS.train is True:
        train(G_train_step, G_loss, D_train_step, D_loss, D_extra_step, G_extra_step, save_img_every=25, print_every=1, max_iter=1000000)
    else:
        test(noise_input, FLAGS.output)


if __name__ == "__main__":
    main()
