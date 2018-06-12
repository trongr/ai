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


def loadImage(path, x_dim):
    """Returns img ~ (1, x_dim), e.g. [[-0.1 . . . 0.9]]"""
    img = np.reshape(np.array(misc.imread(path)), [-1, x_dim])
    return preprocess_img_rgb(img)


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def saveImages(outputDir, images, img_w, img_h, img_c, it):
    fig = plt.figure(figsize=(10.0 * img_w / img_h, 10))
    gs = gridspec.GridSpec(1, 1) if len(images) is 1 else gridspec.GridSpec(10, 10)  # In test mode we just have one image
    gs.update(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        img = deprocess_img(img)
        plt.imshow(img.reshape([img_h, img_w, img_c]))

    output = it if isinstance(it, str) else str(it).zfill(10)
    extension = "" if output.endswith(".jpg") else ".jpg"
    trailingSlash = "" if outputDir.endswith("/") else "/"
    imgpath = outputDir + trailingSlash + output + extension
    print("Saving img: " + imgpath)
    fig.savefig(imgpath)
    plt.close(fig)


def saveEncoding(outputDir, encoding, output):
    output = output if isinstance(output, str) else str(output).zfill(10)
    extension = "" if output.endswith(".txt") else ".txt"
    trailingSlash = "" if outputDir.endswith("/") else "/"
    filename = outputDir + trailingSlash + output + extension
    print("Saving encoding: " + filename)
    np.savetxt(filename, encoding)


def GenerateSimilarEncodings(encoding, count):
    """
    Encoding is a list of floats between -1 and 1. Return a list of similar
    encodings to encoding of length count. The first element in this list is
    encoding. Similar here means we take a few elements of encoding at random,
    and add a small random noise to it, also ensuring that the new values are
    between -1 and 1.
    """
    # TODO. make this method accept the std, so you can narrow down on a face.
    noise = np.random.normal(0, 0.2, size=[count, len(encoding)])
    encodings = np.array([encoding, ] * count)  # Duplicate encoding along height
    encodings = np.clip(encodings + noise, -1, 1)
    encodings[0] = encoding
    return encodings
