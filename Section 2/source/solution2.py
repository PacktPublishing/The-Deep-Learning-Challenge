# Alternative solutions:
# - Get classes names from:
#   ~/.keras/models/imagenet_class_index.json
# - Get classes at http://image-net.org/challenges/LSVRC/2014/browse-synsets
import conf
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import decode_predictions
import numpy as np
from pprint import pprint

if __name__ == '__main__':
    model = VGG19(classes=1000)
    classes=decode_predictions(np.ones((1,1000), dtype=float), top=1000)
    cnames=[ c[1] for c in classes[0]]
    cnames.sort()
    pprint(cnames)
    
