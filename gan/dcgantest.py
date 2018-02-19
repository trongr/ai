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
noise_dim = 96

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

def preprocess_img(x):
    return 2 * x - 1.0

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
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        input_layer = tf.reshape(x, [-1, 28, 28, 1])
        c1 = tf.layers.conv2d(inputs=input_layer, filters=32, 
                kernel_size=5, strides=1, padding='same', 
                activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2)
        c2 = tf.layers.conv2d(inputs=p1, filters=64, kernel_size=5, strides=1, 
                padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2)
        f1 = tf.reshape(p2, [-1, 7 * 7 * 64]) # For same padding
        # f1 = tf.reshape(p2, [-1, 4 * 4 * 64]) # For valid padding
        fc1 = tf.layers.dense(inputs=f1, units=1024, activation=leaky_relu)
        logits = tf.layers.dense(inputs=fc1, units=1)
        return logits

def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
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
        ct1_filter = tf.ones((4, 4, in_channels_2, in_channels_1),
                        dtype=tf.float32)
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
        img = tf.tanh(tf.nn.conv2d_transpose(
                bn3,
                filter=ct2_filter,
                output_shape=(batch_size, 28, 28, 1),
                strides=(1, 2, 2, 1),
                padding='SAME',
                data_format='NHWC'))

        return img

def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
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

def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.
    
    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    
          
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    D_loss = 0.5 * tf.reduce_mean((score_real - 1) ** 2) \
           + 0.5 * tf.reduce_mean(score_fake ** 2)
    G_loss = 0.5 * tf.reduce_mean((score_fake - 1) ** 2)
    return D_loss, G_loss

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
G_sample = generator(z)

with tf.variable_scope("") as scope:
    logits_real = discriminator(preprocess_img(x))
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    logits_fake = discriminator(G_sample)

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

beta1 = 0.5
D_solver = tf.train.AdamOptimizer(learning_rate=5e-7, beta1=beta1)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1)
# D_loss, G_loss = gan_loss(logits_real, logits_fake)
D_loss, G_loss = lsgan_loss(logits_real, logits_fake)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
with tf.control_dependencies(D_extra_step):
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)

G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
with tf.control_dependencies(G_extra_step):
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, 
        D_extra_step, save_img_every=250, print_every=50, batch_size=128, 
        num_epoch=10):
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

    max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
    for it in range(max_iter):
        xmb, _ = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xmb})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        if it % save_img_every == 0:
            samples = sess.run(G_sample)
            save_images(out_dir, samples[:49], it)

        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        if it % 10 == 0:
            Saver.save(sess, save_dir + "/gan", global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, 
        G_extra_step, D_extra_step, save_img_every=200, num_epoch=1000)
