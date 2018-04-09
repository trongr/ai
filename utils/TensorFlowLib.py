import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def getTrainSteps(D_loss, G_loss):
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
    return D_train_step, G_train_step, D_extra_step, G_extra_step
