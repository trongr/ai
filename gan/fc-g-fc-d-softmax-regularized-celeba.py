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
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)
import utils

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

batch_size = 100
img_h = 218
img_w = 178
img_c = 3
w_to_h = 1.0 * img_w / img_h
x_dim = 116412 # 218, 178, 3 dimension of each image
noise_dim = 96

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
    fig = plt.figure(figsize=(10 * w_to_h, 10))
    gs = gridspec.GridSpec(10, 10)
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
    with tf.variable_scope("discriminator"):
        fc1 = tf.contrib.layers.fully_connected(
            x, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        fc2 = tf.contrib.layers.fully_connected(
            fc1, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        fc3 = tf.contrib.layers.fully_connected(
            fc2, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        fc4 = tf.contrib.layers.fully_connected(
            fc3, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        logits = tf.contrib.layers.fully_connected(
            fc4, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        return logits

def generator(z, keep_prob):
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(
            z, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        fc2 = tf.contrib.layers.fully_connected(
            fc1, num_outputs=64,
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

        fc3 = tf.contrib.layers.fully_connected(
            fc2, num_outputs=64,
             activation_fn=leaky_relu,
             weights_initializer=tf.contrib.layers.xavier_initializer(),
             biases_initializer=tf.constant_initializer(0.1),
             trainable=True)

        img = tf.contrib.layers.fully_connected(
            fc3, num_outputs=x_dim,
            activation_fn=tf.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True)

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

def Discriminator_Regularizer(D_real, x, D_fake, G_sample):
    D1 = tf.nn.sigmoid(D_real)
    D2 = tf.nn.sigmoid(D_fake)
    grad_D_real = tf.gradients(D_real, x)[0]
    grad_D_fake = tf.gradients(D_fake, G_sample)[0]
    grad_D_real_norm = tf.norm(tf.reshape(grad_D_real, [batch_size, -1]), axis=1, keep_dims=True)
    grad_D_fake_norm = tf.norm(tf.reshape(grad_D_fake, [batch_size, -1]), axis=1, keep_dims=True)

    reg_D1 = tf.multiply(tf.square(1.0 - D1), tf.square(grad_D_real_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D_fake_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer

tf.reset_default_graph()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x-input")
    z = tf.placeholder(tf.float32, shape=[None, noise_dim], name="z-input")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope("") as scope:
    G_sample = generator(z, keep_prob)
    D_real = discriminator(x)
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    D_fake = discriminator(G_sample)

# D_loss, G_loss = gan_loss(D_real, D_fake)

D_target = 1. / batch_size
G_target = 1. / batch_size
Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))

# Apply regularization on D loss
gamma = 0.1
D_regu = gamma / 2.0 * Discriminator_Regularizer(D_real, x, D_fake, G_sample)
D_loss = tf.reduce_sum(D_target * D_real) + log(Z) + D_regu

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
prefix = "fc-g-fc-d-softmax"
save_dir = "save"
save_dir_prefix = save_dir + "/" + prefix
logs_path = "logs/" + prefix
mkdir_p(out_dir)
mkdir_p(save_dir)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

def train(sess, G_train_step, G_loss, D_train_step, D_loss,
    D_extra_step, G_extra_step,
    save_img_every=250, print_every=50, max_iter=1000000):
    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    batches = load_images(img_dir)
    t = time.time()
    for it in range(max_iter):
        # xmb = batches.next() # TODO. Remove
        xmb = next(batches)
        z_noise = sample_z(batch_size, noise_dim)

        if it % save_img_every == 0:
            samples = sess.run(G_sample, feed_dict={
                x: xmb, z: z_noise, keep_prob: 1.0
            })
            save_images(out_dir, samples[:100], it)

        # train G twice for every D train step. see if that helps learning.
        _, D_loss_curr, summary, _ = sess.run([
            D_train_step, D_loss, summary_op, D_extra_step
        ], feed_dict={
            x: xmb, z: z_noise, keep_prob: 0.3
        })
        _, G_loss_curr, _ = sess.run([
            G_train_step, G_loss, G_extra_step
        ], feed_dict={
            x: xmb, z: z_noise, keep_prob: 0.3
        })
        _, G_loss_curr, _ = sess.run([
            G_train_step, G_loss, G_extra_step
        ], feed_dict={
            x: xmb, z: z_noise, keep_prob: 0.3
        })

        if math.isnan(D_loss_curr) or math.isnan(G_loss_curr):
            print("D or G loss is nan", D_loss_curr, G_loss_curr)
            exit()

        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}, Elapsed: {:.4}'
                .format(it, D_loss_curr, G_loss_curr, time.time() - t))
            t = time.time()

        if it % 10 == 0:
            writer.add_summary(summary, global_step=it)
            Saver.save(sess, save_dir_prefix, global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    train(sess, G_train_step, G_loss, D_train_step, D_loss,
        D_extra_step, G_extra_step,
        save_img_every=25, print_every=1, max_iter=1000000)