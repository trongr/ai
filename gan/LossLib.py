import tensorflow as tf


def VanillaGANLoss(D_real, D_fake):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_fake), logits=D_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real), logits=D_real)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake), logits=D_fake))
    return D_loss, G_loss


def LeastSquaresLoss(score_real, score_fake):
    D_loss = 0.5 * tf.reduce_mean((score_real - 1) ** 2) + 0.5 * tf.reduce_mean(score_fake ** 2)
    G_loss = 0.5 * tf.reduce_mean((score_fake - 1) ** 2)
    return D_loss, G_loss


def MeanSquaredDiff(a, b):
    return tf.reduce_mean((a - b) ** 2)


def SoftmaxLoss(D_real, D_fake, batch_size):
    Z = tf.reduce_sum(tf.exp(-D_real)) + tf.reduce_sum(tf.exp(-D_fake))
    D_loss = tf.reduce_sum(1. / batch_size * D_real) + tf.log(Z)
    G_loss = tf.reduce_sum(1. / batch_size * D_fake) + tf.reduce_sum(1. / batch_size * D_real) + log(Z)
    return D_loss, G_loss


def WGANLoss(discriminator, D_real, D_fake, x, G_sample, batch_size, LAMBDA=10):
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
