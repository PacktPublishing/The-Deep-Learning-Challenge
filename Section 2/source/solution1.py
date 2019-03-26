import conf
from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    model=VGG16()
    print(model)
