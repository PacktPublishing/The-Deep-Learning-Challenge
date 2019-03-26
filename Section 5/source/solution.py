"""
Language translator based on:
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
"""
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

# Training parameters
batch_size = 64
epochs = 50

# 1. Data preparation.
num_samples = 10000
data_path = 'spa-eng/spa.txt'

# Vectorize the data.
input_texts = []
target_texts = []

# Two character sets that we will use
# first to turn our text into vectors,
# then let encoder and decoder know
# how much data do we have and in the end
# to decode the output from our sampling/prediciton
# models.
input_characters = set()
target_characters = set()

# Read data and put into input and target sets.
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# Since we will be mapping our character sets
# to vectors we want them to be sorted.
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# Get the necessary information for both encoder and decoder
# about the number of characters they will both handle.
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# Find what's the longest sentence in both languages.
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Create a mapping between the characters in each set and their indexes
# We will use that to create vectors in the moment and later to decode
# the output data from our prediction/sampling model.
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Create our empty one-hot vectors.
encoder_input_data = np.zeros(
(len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

decoder_input_data = np.zeros(
(len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
(len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

print('Encoder input data shape', encoder_input_data.shape)
print('Decoder input data shape', decoder_input_data.shape)

# Populate each vector according to mapping in sets.
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# 2. Model creation.
# Latent dimensionality of the encoding space.
latent_dim = 256

# 2.1 Create encoder.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# 2.2 Create decoder with input from encoder's state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well.
# We don't use the return states in the training model,
# but we will use them in prediction/sampling phase.
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# encoder_input_data and decoder_input_data into decoder_target_data
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

# 3. Training.

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# 3. Prediction/testing/sampling.
#

# Define prediction/sampling models
# using previously trained parts of models.
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder(decoder_inputs,
                                                 initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                      outputs=[decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
(i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
(i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # Feed the decoder to get the next characters using encoder's state.
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the the location of the most probable character
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # Get the next character.
        sampled_char = reverse_target_char_index[sampled_token_index]
        # Add it to sentence.
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Take one sequence (part of the training set)
# for trying out decoding.
for seq_index in range(100):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index].encode('ascii','ignore'))
    print('Decoded sentence:', decoded_sentence.encode('ascii','ignore'))
