import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
# Helper functions to deal with image preprocessing
from cs231n.image_utils import load_image, preprocess_image, deprocess_image
from cs231n.classifiers.squeezenet import SqueezeNet
import tensorflow as tf

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

tf.reset_default_graph() # remove all existing variables in the graph 
sess = get_session() # start a new Session

# Load pretrained SqueezeNet model. Model came with split save files, so we
# check for one and use the common file name
SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt.meta'
if not os.path.exists(SAVE_PATH):
    raise ValueError("You need to download SqueezeNet!")
else:
    SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'        
    model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

# Load data for testing
content_img_test = preprocess_image(load_image('styles/tubingen.jpg', size=192))[None]
style_img_test = preprocess_image(load_image('styles/starry_night.jpg', size=192))[None]
answers = np.load('style-transfer-checks-tf.npz')
print(answers) # <numpy.lib.npyio.NpzFile object at 0x00000092F73B6DD8>

def reduce_sum_squared_difference(a, b):
    return tf.reduce_sum(tf.squared_difference(a, b))

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    r = reduce_sum_squared_difference(content_current, content_original)
    return tf.multiply(content_weight, r)

def content_loss_test(correct):
    content_layer = 3
    content_weight = 6e-2
    c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img_test})
    bad_img = tf.zeros(content_img_test.shape)
    feats = model.extract_features(bad_img)[content_layer]
    student_output = sess.run(content_loss(content_weight, c_feats, feats))
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))

content_loss_test(answers['cl_out'])

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    shape = tf.shape(features)
    h, w, c = shape[1], shape[2], shape[3]
    n = tf.cast(h * w * c, tf.float32)
    f = tf.reshape(features, [-1, c])
    ft = tf.transpose(f)
    gram = tf.matmul(ft, f)
    if normalize:
        gram = tf.divide(gram, n)
    return gram
    
def gram_matrix_test(correct):
    gram = gram_matrix(model.extract_features()[5])
    student_output = sess.run(gram, {model.image: style_img_test})
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))

gram_matrix_test(answers['gm_out'])

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as
      produced by the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to
      include in the style loss.
    - style_targets: List of the same length as style_layers, where
      style_targets[i] is a Tensor giving the Gram matrix the source style image
      computed at layer style_layers[i].
    - style_weights: List of the same length as style_layers, where
      style_weights[i] is a scalar giving the weight for the style loss at layer
      style_layers[i].

    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    L = tf.Variable(0.0)
    sess.run(L.initializer)    
    for i, l in enumerate(style_layers):
        G = gram_matrix(feats[l], normalize=True)
        A = style_targets[i]
        w = tf.cast(style_weights[i], tf.float32)
        r = reduce_sum_squared_difference(G, A)
        Ll = tf.multiply(w, r)
        L = tf.assign_add(L, Ll)        
    return L

def style_loss_test(correct):
    style_layers = [1, 4, 6, 7]
    style_weights = [300000, 1000, 15, 3]
    
    feats = model.extract_features()
    style_target_vars = []
    for idx in style_layers:
        style_target_vars.append(gram_matrix(feats[idx]))
    style_targets = sess.run(style_target_vars,
                             {model.image: style_img_test})
                             
    s_loss = style_loss(feats, style_layers, style_targets, style_weights)
    student_output = sess.run(s_loss, {model.image: content_img_test})
    error = rel_error(correct, student_output)
    print('Error is {:.3f}'.format(error))

style_loss_test(answers['sl_out'])