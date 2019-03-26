#!/usr/bin/env python3
from os import listdir
from os import path

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pprint import pprint

def gen_x(xtext, tokenizer, max_len=None, for_training=False):
    # We "fit" our tokenizer on our training set.
    # This is where unique numbers are generated for each word.
    if for_training:
        tokenizer.fit_on_texts(xtext)

    # Encode words(tokens) as unique numbers.
    encoded_xtext = tokenizer.texts_to_sequences(xtext)

    # We're looking for the longest sentence
    # in our training set.
    # Then we will use it when we ran gen_x on test data.
    # The key here is to have maximum lenght all the same trougout
    # training and data sets.
    if not max_len:
        max_len = max([len(s.split()) for s in xtext])
        tokenizer._max_padding_len=max_len

    # We need to pad our encoded text to the maximum lenght
    # for our embedding layer to work properly.
    train_x = pad_sequences(encoded_xtext, maxlen=max_len, padding='post')

    if for_training:
        return train_x, max_len
    return train_x


def get_data(d='review_polarity/txt_sentoken'):
    train_x=[]
    train_y=[]

    test_x=[]
    test_y=[]

    # First, load all of the data into train_x.
    print('Loading data...')
    for p in ['neg', 'pos']:
        for filename in listdir(path.join(d,p)):
            dfile = path.join(d,p,filename)
            data=open(dfile).read()
            train_x.append(data)

    ct=train_x

    # Split our data set as training and test set.
    # We have 1000 positive and 100 negative reviews.
    l=1000
    # We split our data into 90% of data for training set
    # and we leave 10% for testing.
    trainl=int(l*0.90)
    testl=int(l*0.10)

    # First, spliting training set.
    # Negative first.
    train_x_neg=ct[0:trainl]
    train_x_pos=ct[l:l+trainl]

    # Generate approriate labels for negative data.
    # 0 means negative, 1 positive.
    train_y_neg=[ 0 for i in range(len(train_x_neg))]
    train_y_pos=[ 1 for i in range(len(train_x_pos))]

    # Put all of training splits together.
    train_x=train_x_neg+train_x_pos
    train_y=train_y_neg+train_y_pos

    # Get the remining 10% of data as test set.
    test_x_neg=ct[trainl:l]
    test_x_pos=ct[l+trainl:]

    test_y_neg=[ 0 for i in range(len(test_x_neg))]
    test_y_pos=[ 1 for i in range(len(test_x_neg))]

    test_x=test_x_neg+test_x_pos
    test_y=test_y_neg+test_y_pos

    # Create a new tokenizer, we will use it for both
    # training and test data.
    tokenizer=Tokenizer()
    # Encode and pad our train and test data.
    input_train_x=train_x
    train_x, max_len=gen_x(train_x, tokenizer, for_training=True)
    test_x=gen_x(test_x, tokenizer, max_len=max_len)

    # Get a vocabulary size (a number of unique words).
    # We will later have to use it for our Embedding layer.
    inputs = len(tokenizer.word_index) + 1

    return train_x, train_y, test_x, test_y, inputs, max_len


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, inputs, max_len, t=get_data()
    print('X[0]', train_x[0])
    print('Y[0]', train_y[0])
