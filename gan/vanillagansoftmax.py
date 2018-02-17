from __future__ import print_function, division
import os
import glob
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
x_dim = 784 # 28 x 28
z_dim = 96

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
    return tf.random_uniform([batch_size, dim], minval=-1, maxval=1, dtype=tf.float32)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, x_dim]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        fc1 = tf.contrib.layers.fully_connected(
            x, num_outputs=256, 
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        fc2 = tf.contrib.layers.fully_connected(
            fc1, num_outputs=256, 
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        logits = tf.contrib.layers.fully_connected(
            fc2, num_outputs=1, 
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        return logits

def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, z_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, x_dim].
    """
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(
            z, num_outputs=1024, 
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        fc2 = tf.contrib.layers.fully_connected(
            fc1, num_outputs=1024, 
            activation_fn=leaky_relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        img = tf.contrib.layers.fully_connected(
            fc2, num_outputs=x_dim, 
            activation_fn=tf.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        return img

def log(x):
    return tf.log(x + 1e-8)

def softmax_gan_loss(D_real, D_fake):
    """
    Inputs:
    - D_real: Tensor, shape [batch_size, 1], output of discriminator
        probability that the image is real for each real image
    - D_fake: Tensor, shape[batch_size, 1], output of discriminator
        probability that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    D_target = 1. / batch_size
    G_target = 1. / batch_size
    Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))
    D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
    G_loss = tf.reduce_sum(G_target * D_fake) + log(Z)
    return D_loss, G_loss

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, x_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
G_sample = generator(z)

with tf.variable_scope("") as scope:
    D_real = discriminator(preprocess_img(x))
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    D_fake = discriminator(G_sample)

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

D_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5)
D_loss, G_loss = softmax_gan_loss(D_real, D_fake)

D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, 
        D_extra_step, save_img_every=250, print_every=50, batch_size=128, 
        num_epoch=10):
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
        zmb = sample_z(batch_size, z_dim)         
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xmb, z: zmb})
        _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xmb, z: zmb})

        if it % save_img_every == 0:
            samples = sess.run(G_sample, feed_dict={x: xmb, z: zmb})
            save_images(out_dir, samples[:49], it)

        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        if it % 10 == 0:
            Saver.save(sess, save_dir + "/gan", global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, 
        G_extra_step, D_extra_step, save_img_every=200, num_epoch=1000)
