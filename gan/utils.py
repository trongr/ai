import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'


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


def load_images(batch_size, x_dim, img_dir, total=None):
    """
    Pass in total=10000 to train only on the first 10000 training points.
    Hopefully training a network on a smaller dataset in the beginning will help
    it converge faster, much like learning in small chunks instead of all at
    once helps in human learning.
    """
    img_paths = []
    for img in os.listdir(img_dir):
        img_paths.append(os.path.join(img_dir, img))
    np.random.shuffle(img_paths)
    print("First 10 images", img_paths[:10])
    if total is None:
        total = len(img_paths)
    i = 0
    while (True):
        if i + batch_size >= total:
            i = 0
            continue
        images = []
        for j in range(batch_size):
            images.append(misc.imread(img_paths[i + j]))
        images = np.reshape(np.asarray(images), [-1, x_dim])
        images = preprocess_img_rgb(images)
        assert not np.any(np.isnan(images)), "Images should not contain nan's"
        yield(images)
        i = (i + batch_size) % total


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_images(dir, images, img_w, img_h, img_c, it):
    fig = plt.figure(figsize=(10.0 * img_w / img_h, 10))
    if len(images) is 1:  # In test mode we just have one image
        gs = gridspec.GridSpec(1, 1)
    else:
        gs = gridspec.GridSpec(10, 10)
    gs.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        img = deprocess_img(img)
        plt.imshow(img.reshape([img_h, img_w, img_c]))

    imgpath = dir + "/" + str(it).zfill(10) + ".jpg"
    print("Saving img " + imgpath)
    fig.savefig(imgpath)
    plt.close(fig)
