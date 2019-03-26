import conf
import numpy as np
from solution1 import get_all
from solution2 import get_mnist

def get_noise(n):
    return np.random.normal(0, 1, size=[n, 100])

def train_generator(gan, discriminator, batch_size):
      """
      Train generator with noise and label
      it as real data.
      """
      X = get_noise(batch_size)
      # Label noise as real data meaning as 1.0.
      Y = np.ones(batch_size)
      # Freeze discriminator to train generator only.
      discriminator.trainable = False
      gan.train_on_batch(X, Y)

def train_discriminator(gan, generator, discriminator, x_train, batch_size):
      """
      Train discriminator with both fake and real data,
      in both cases we provide correct informations about
      the data.
      """
      # Get a random set of input noise.
      noise = get_noise(batch_size)

      # Generate fake MNIST images.
      generated_images = generator.predict(noise)
      # Get a random set of images from the actual MNIST dataset.
      image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
      # Put them together in a single vector(list).
      X = np.concatenate([image_batch, generated_images])

      # Generate 0.0 (fake) for the whole vector.
      Y = np.zeros(2*batch_size)
      # Label real images correctly as 1.0.
      Y[:batch_size] = 1.0

      discriminator.trainable = True
      discriminator.train_on_batch(X, Y)


def train(epochs, batch_size, save_gan_images=None):
      """
      Train our GAN for a number of epochs (training sessions)
      and using batch_size samples in each epoch.

      Save a snapshot of learned mnist images at even epochs.
      """
      x_train = get_mnist()
      gan, generator, discriminator = get_all()

      batch_count = x_train.shape[0] / batch_size

      for e in range(1, epochs+1):
          for i in range(int(batch_count)):
              print('Epoch %d of %d [ batch %d of %d ]' % (e, epochs, i+1, batch_count), end='\r', flush=True)
              train_discriminator(gan, generator, discriminator, x_train, batch_size)
              train_generator(gan, discriminator, batch_size)

          if e == 1 or e % 2 == 0:
              if save_gan_images:
                  save_gan_images(generator, e)

if __name__ == '__main__':
    train(50, 256)
