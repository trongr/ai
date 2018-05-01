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
