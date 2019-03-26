import conf
from sys import argv
from keras.applications.inception_v3 import InceptionV3
# You need to install pillow package for load_img:
# conda install load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

from pprint import pprint

# load the model
model = InceptionV3()

print(argv[1])

# Load image from the first script's argument.
# Resise to fit model's required input.
image = load_img(argv[1], target_size=(299, 299))

# Convert image to numpy array.
image = img_to_array(image)

# Reshape image to fit the model's requirments.
# Fist argument is the number of images we plan
# to classify using the model.
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# Prepare the image in the same way
# that images that the model was trained on.
image = preprocess_input(image)

print(image)
