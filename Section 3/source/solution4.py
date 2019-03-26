from solution1 import get_model
from solution2 import get_data
from keras.callbacks import TensorBoard
# Run tensorboard --logdir /tmp/tb to see the chart.

def train_model(batch_size, epochs, inputs, max_length, train_x, train_y, test_x, test_y):
    """
    Compile and train model with choosen parameters.
    """
    model=get_model(inputs, max_length)
    # Compile model
    # Fit model on training data, validate during training on test data.
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
    return model

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, inputs, max_len=get_data()
    model = train_model(64, 10, inputs, max_len, train_x, train_y, test_x, test_y)
    # Test our model on both data that has been seen
    # (training data set) and unseen (test data set)
    print('Evaluation:')
    loss, acc = model.evaluate(train_x, train_y, verbose=2)
    print('Train Accuracy: %.2f%%' % (acc*100))
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    print('Test Accuracy: %.2f%%' % (acc*100))
