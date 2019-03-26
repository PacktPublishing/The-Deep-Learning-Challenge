import conf
from keras.datasets import mnist
import numpy as np

def get_mnist():
    # Loading mnist data, we're using only training data.
    (x_train, _), _ = mnist.load_data()
    # Image vector size used as output size for Generator
    # and input size for Discriminator.
    # In case of MNIST data set we create we have a vector of size 28*28.
    ivs=x_train.shape[1]*x_train.shape[2]
    # Normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # Reshape to match our Discriminator input size.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    return x_train

if __name__ == '__main__':
    data=get_mnist()
    print(data[0])
