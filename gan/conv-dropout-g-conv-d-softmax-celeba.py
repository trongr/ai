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

batch_size = 128
img_w = 28
img_h = 28
img_c = 1
w_to_h = 1.0 * img_w / img_h
x_dim = 784 # 218, 178, 3 dimension of each image
noise_dim = 96
# batch_size = 128
# img_w = 178
# img_h = 218
# img_c = 3
# w_to_h = 1.0 * img_w / img_h
# x_dim = 116412 # 218, 178, 3 dimension of each image
# noise_dim = 96

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
        images = np.reshape(np.asarray(images), [batch_size, -1])
        yield(images)
        i = (i + batch_size) % total

def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_images(dir, images, it):
    images = np.reshape(images, [images.shape[0], -1]) # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(10 * w_to_h, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        # poij
        # plt.imshow(img.reshape([img_h, img_w, img_c]))
        plt.imshow(img.reshape([img_h, img_w]))

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
    with tf.variable_scope("discriminator"):
        fc1 = tf.contrib.layers.fully_connected(
            x, num_outputs=64, 
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
            activation_fn=tf.tanh,
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

        img = tf.contrib.layers.fully_connected(
            fc1, num_outputs=x_dim, 
            activation_fn=tf.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            trainable=True) 

        return img    

def log(x):
    return tf.log(x + 1e-8)

def wgangp_loss(D_real, D_fake, x, G_sample):
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

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x-input")
    z = tf.placeholder(tf.float32, shape=[None, noise_dim], name="z-input")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.variable_scope("") as scope:
    G_sample = generator(z, keep_prob)    
    # poij
    # D_real = discriminator(preprocess_img(x)) # scale images to be -1 to 1
    D_real = discriminator(x) # scale images to be -1 to 1
    scope.reuse_variables() # Re-use discriminator weights on new inputs        
    D_fake = discriminator(G_sample)

# D_loss, G_loss = wgangp_loss(D_real, D_fake, x, G_sample)
D_target = 1. / batch_size
G_target = 1. / batch_size
Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))
D_loss = tf.reduce_sum(D_target * D_real) + log(Z)
G_loss = tf.reduce_sum(G_target * D_fake) + log(Z)

tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()

dlr, glr, beta1 = 1e-3, 1e-3, 0.5
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_solver = tf.train.AdamOptimizer(learning_rate=dlr, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=glr, beta1=beta1)    
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

img_dir = "./data/img_align_celeba/"
out_dir = "out"
prefix = "conv-dropout-g-conv-d-softmax"
save_dir = "save"
save_dir_prefix = save_dir + "/" + prefix
logs_path = "logs/" + prefix
mkdir_p(out_dir)
mkdir_p(save_dir)

writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

def train(sess, G_train_step, G_loss, D_train_step, D_loss,
              show_every=250, print_every=50, max_iter=1000000):
    Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1) 
    if glob.glob(save_dir + "/*"):
        Saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    t = time.time()        
    for it in range(max_iter):
        xmb, _ = mnist.train.next_batch(batch_size)
        z_noise = sample_z(batch_size, noise_dim)   

    # batches = load_images(img_dir)
    # t = time.time()        
    # for it in range(max_iter):
    #     xmb = batches.next()
    #     z_noise = sample_z(batch_size, noise_dim)          

        if it % show_every == 0:
            samples = sess.run(G_sample, feed_dict={
                x: xmb, z: z_noise, keep_prob: 1.0
            })
            save_images(out_dir, samples[:100], it)

        _, D_loss_curr, summary = sess.run([D_train_step, D_loss, summary_op], 
            feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})
        _, G_loss_curr = sess.run([G_train_step, G_loss], 
            feed_dict={x: xmb, z: z_noise, keep_prob: 0.3})

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
        show_every=50, max_iter=1000000)
