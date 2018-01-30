import tensorflow as tf

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

tf.reset_default_graph() # remove all existing variables in the graph 
sess = get_session() # start a new Session

def loss():
    L = tf.Variable(0.0)
    M = tf.Variable(1.0)    
    sess.run(L.initializer)
    sess.run(M.initializer)
    L = tf.assign(L, 0.0)
    for i in range(10):
        L = tf.assign_add(L, 1.0)
        # L += Ll
    return L

# What the heck. This does the right thing.
L = loss()
print(sess.run(L))
print(sess.run(L))