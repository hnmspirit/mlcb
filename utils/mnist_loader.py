import numpy as np


def _load_mnist_images(file_name, verbose=True):
    with open(file_name, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        nitem = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        heigh = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        width = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        image = np.frombuffer(f.read(nitem*heigh*width), dtype=np.uint8).newbyteorder()
        image = image.reshape(nitem, heigh, width)
    if verbose: print('loaded mnist images (id={}): {}'.format(magic, image.shape))
    return image


def _load_mnist_labels(file_name, verbose=True):
    with open(file_name, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        nitem = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder().item()
        label = np.frombuffer(f.read(nitem), dtype=np.uint8).newbyteorder()
    if verbose: print('loaded mnist labels (id={}): {}'.format(magic, label.shape))
    return label


def load_mnist(image_file_name, label_file_name, verbose=True):
    images = _load_mnist_images(image_file_name, verbose)
    labels = _load_mnist_labels(label_file_name, verbose)
    return images, labels