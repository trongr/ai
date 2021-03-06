from __future__ import print_function, division
import os
import tensorflow as tf
import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
x_dim = 784 # 28 * 28, dimension of each image
# noise_dim = 96 # TODO. Adjust
noise_dim = 32

def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_images(dir, images, it):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    imgpath = dir + "/" + str(it).zfill(10) + ".jpg"
    print("Saving img " + imgpath)
    fig.savefig(imgpath)
    plt.close(fig)
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(alpha * x, x)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    return tf.random_uniform([batch_size, dim], minval=-1, maxval=1,
                             dtype=tf.float32)

def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, x_dim]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        input_layer = tf.reshape(x, [-1, 28, 28, 1])
        c1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                kernel_size=5, strides=1, padding='same',
                activation=leaky_relu)
        c2 = tf.layers.conv2d(inputs=c1, filters=64, kernel_size=5, strides=1,
                padding='same', activation=leaky_relu)
        f1 = tf.reshape(c2, [-1, 7 * 7 * 64])
        fc1 = tf.layers.dense(inputs=f1, units=1024, activation=leaky_relu)
        logits = tf.layers.dense(inputs=fc1, units=1)
        return logits

def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, x_dim].
    """
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(z, num_outputs=1024,
                activation_fn=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(
                fc1,
                axis=-1,
                momentum=0.99, # use default
                epsilon=0.001, # use default
                center=True, # enable beta
                scale=True, # enable gamma
                training=True)
        # Adversarial is always training, unless you're using generator to
        # generate images (which you can also do during training).

        in_channels_1 = 128
        fc2 = tf.contrib.layers.fully_connected(bn1,
                num_outputs=7 * 7 * in_channels_1,
                activation_fn=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(
                fc2,
                axis=-1,
                momentum=0.99, # use default
                epsilon=0.001, # use default
                center=True, # enable beta
                scale=True, # enable gamma
                training=True)
        rs1 = tf.reshape(bn2, [-1, 7, 7, in_channels_1])

        in_channels_2 = 64
        ct1_filter = tf.get_variable(
                        "ct1_filter",
                        shape=(4, 4, in_channels_2, in_channels_1),
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(),
                        trainable=True)
        ct1 = tf.nn.relu(tf.nn.conv2d_transpose(
                rs1,
                filter=ct1_filter,
                output_shape=(batch_size, 14, 14, in_channels_2),
                strides=(1, 2, 2, 1),
                padding='SAME',
                data_format='NHWC'))
        bn3 = tf.layers.batch_normalization(
                ct1,
                axis=-1, # batch normalize over channels (alt: axis=3)
                momentum=0.99, # use default
                epsilon=0.001, # use default
                center=True, # enable beta
                scale=True, # enable gamma
                training=True)

        ct2_filter = tf.get_variable(
                        "ct2_filter",
                        shape=(4, 4, 1, in_channels_2),
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(),
                        trainable=True)
        ct2 = tf.nn.relu(tf.nn.conv2d_transpose(
                bn3,
                filter=ct2_filter,
                output_shape=(batch_size, 28, 28, 1),
                strides=(1, 2, 2, 1),
                padding='SAME',
                data_format='NHWC'))

        avg = tf.reduce_mean(ct2, axis=3, keep_dims=True)
        rs2 = tf.reshape(avg, [-1, x_dim]) # Reshape cause we want ~ (N, 784) instead of ~ (N, 28, 28, 1)
        # Need this last FC layer cause for some reason you can't have reshape
        # as the last layer cause it has no gradient.
        img = tf.contrib.layers.fully_connected(rs2, num_outputs=x_dim,
                activation_fn=tf.tanh)

        return img

def log(x):
    return tf.log(x + 1e-8)

def wgangp_loss(D_real, D_fake, x, G_sample):
    """
    Compute the WGAN-GP loss.

    Inputs:
    - D_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - D_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    - x: the input (real) images for this batch
    - G_sample: the generated (fake) images for this batch

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    LAMBDA = 10
    G_loss = -tf.reduce_mean(D_fake)
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

    alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
    differences = G_sample - x
    interpolates = x + (alpha * differences)

    with tf.variable_scope('', reuse=True) as scope:
        gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

    D_loss += LAMBDA * gradient_penalty

    return D_loss, G_loss

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, x_dim])
z = tf.placeholder(tf.float32, shape=[None, noise_dim])
G_sample = generator(z)

with tf.variable_scope("") as scope:
    # scale images to be -1 to 1
    D_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    D_fake = discriminator(G_sample)

D_loss, G_loss = wgangp_loss(D_real, D_fake, x, G_sample)
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

dlr, glr, beta1 = 1e-3, 1e-3, 0.5
D_solver = tf.train.AdamOptimizer(learning_rate=dlr, beta1=beta1)
G_solver = tf.train.AdamOptimizer(learning_rate=glr, beta1=beta1)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
with tf.control_dependencies(D_extra_step):
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
with tf.control_dependencies(G_extra_step):
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=250, print_every=50, batch_size=128, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    out_dir = "out"
    save_dir = "save"
    mkdir_p(out_dir)
    mkdir_p(save_dir)

    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    # compute the number of iterations we need
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    for it in range(max_iter):
        xmb, _ = mnist.train.next_batch(batch_size)
        z_noise = sample_z(batch_size, noise_dim)

        if it % show_every == 0:
            samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise})
            save_images(out_dir, samples[:49], it)

        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xmb, z: z_noise})
        _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xmb, z: z_noise})

        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        if it % 10 == 0:
            Saver.save(sess, save_dir + "/gan", global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss,
            G_extra_step, D_extra_step, show_every=200, num_epoch=1000)
