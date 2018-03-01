from __future__ import print_function, division
import os
import time
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
x_dim = 784 # 28 * 28, dimension of each image
noise_dim = 96

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
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2)
        c2 = tf.layers.conv2d(inputs=p1, filters=64, kernel_size=5, strides=1, 
                padding='same', activation=leaky_relu)
        p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2)
        f1 = tf.reshape(p2, [-1, 7 * 7 * 64])
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

        fc2 = tf.contrib.layers.fully_connected(bn1, 
                num_outputs=2 * 2 * 28 * 28, 
                activation_fn=tf.nn.relu) 
        bn2 = tf.layers.batch_normalization(
                fc2,
                axis=-1,
                momentum=0.99, # use default
                epsilon=0.001, # use default
                center=True, # enable beta
                scale=True, # enable gamma
                training=True)
        rs1 = tf.reshape(bn2, [-1, 2 * 28, 2 * 28, 1])

        c1 = tf.layers.conv2d(inputs=rs1, filters=16, 
                kernel_size=5, strides=1, padding='same', 
                activation=leaky_relu)
        p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2) 

        c2 = tf.layers.conv2d(inputs=p1, filters=32, 
                kernel_size=5, strides=1, padding='same', 
                activation=leaky_relu)

        # We don't actually need another layer to turn the (N, 28, 28, 1) img
        # into (N, 784). But verify just to be sure. In any case another layer 
        # might help the network learn.
        #
        # img = tf.reduce_mean(c2, axis=3, keep_dims=True)
        #
        avg = tf.reduce_mean(c2, axis=3, keep_dims=True)
        rs2 = tf.reshape(x, [-1, x_dim]) # Reshape cause we want ~ (N, 784) instead of ~ (N, 28, 28, 1)
        # Need this last FC layer cause for some reason you can't have reshape
        # as the last layer cause it has no gradient.
        img = tf.contrib.layers.fully_connected(rs2, num_outputs=x_dim, 
                activation_fn=tf.tanh) 

        return img 
 
def log(x):
    return tf.log(x + 1e-8)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, x_dim])
z = tf.placeholder(tf.float32, shape=[None, noise_dim])
G_sample = generator(z)

with tf.variable_scope("") as scope:
    D_real = discriminator(preprocess_img(x)) # scale images to be -1 to 1
    scope.reuse_variables() # Re-use discriminator weights on new inputs
    D_fake = discriminator(G_sample)

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

D_target = 1. / batch_size
G_target = 1. / batch_size
Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))
D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
G_loss = tf.reduce_sum(G_target * D_fake) + log(Z)

dlr, glr = 1e-3, 1e-3
beta1 = 0.5
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

        if it % show_every == 0:
            samples = sess.run(G_sample, feed_dict={x: xmb, z: z_noise})
            save_images(out_dir, samples[:49], it)

        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xmb, z: z_noise})
        _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xmb, z: z_noise})

        if it % print_every == 0: # We want to make sure D_loss doesn't go to 0
            print('Iter: {}, D: {:.4}, G: {:.4}, Elapsed: {:.4}'.format(it, D_loss_curr, G_loss_curr, time.time() - t))            
            t = time.time()

        if it % 10 == 0:
            Saver.save(sess, save_dir + "/gan", global_step=it)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, 
            G_extra_step, D_extra_step, show_every=200, num_epoch=1000)
