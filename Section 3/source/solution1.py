import conf

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def get_model(inputs, max_length, dim=25):
    """
    input - vocabulary size, a number of unique words in
            our data set

    max_length - the maximum length of each sequence of words
                 (a document)

    dim - word embedding dimension, the lenght of word vector
          that will be produced by this layer
    """
    model = Sequential()
    model.add(Embedding(inputs, dim, input_length=max_length))
    # Extract feature maps/most common "phrases".
    model.add(Conv1D(filters=25, kernel_size=5, activation='relu'))
    # Pick up the "best ones", pooling==reducting.
    model.add(MaxPooling1D(pool_size=4))
    # Just put everything together into one vector.
    model.add(Flatten())
    # This is the standard output for classification.
    # It matches our two classes 0 and 1.
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model=get_model(10, 20)
    print(model)
