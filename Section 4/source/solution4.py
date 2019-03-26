from solution3 import get_noise, train
import matplotlib.pyplot as plt

def save_gan_images(generator, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    """
    Generate a sample of examples.
    """
    noise = get_noise(examples)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    imgn='results/gan_mnist_%d.png' % epoch
    plt.savefig(imgn)
    return imgn

if __name__ == '__main__':
    train(50, 256, save_gan_images)
