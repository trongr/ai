def preprocess_img_rgb(x):
    """
    For preprocessing RGB images, which are between 0 and 255 instead of 0 and
    1. Sometimes when training we want to make them [-1, 1].
    """
    return x / 255. * 2 - 1

def preprocess_img(x):
    """
    Image values are between 0 and 1, but sometimes when training we want to
    make them [-1, 1].
    """
    return 2.0 * x - 1.0

def deprocess_img(x):
    """
    Convert values of a generated image from [-1, 1] back to [0, 1] for saving.
    """
    return (x + 1.0) / 2.0