## https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import CSVLogger
import numpy as np
import h5py
import pandas
import sys
import os
import random
import string
import json
from argparse import ArgumentParser

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg

parser = ArgumentParser(description="Load character seq2seq model")
parser.add_argument("--epochs", dest="epochs", required=False,
                    help="number of epochs", default=100,
                    type=int)
parser.add_argument("--latent-dim", dest="latent_dim", required=False,
                    help="path to options.json file", default=100,
                    type=int)
parser.add_argument("--clip-length", dest="clip_length", required=False,
                    help="path to dict.json file", default=None,
                    type=int)
args = parser.parse_args()

DATA_DIR = '../data/'
batch_size = 10  # Batch size for training.
epochs = args.epochs  # Number of epochs to train for.
latent_dim = args.latent_dim  # Latent dimensionality of the encoding space.
num_samples = 10000 #10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = DATA_DIR+'embeddings.hdf5'

input_texts, target_texts = [], []
input_characters, target_characters = set(), set()
lemma_file = DATA_DIR+'train.csv'
CLIP_LENGTH = args.clip_length if args.clip_length else 100

OUT_BASE_DIR = '../model'
OUT_DIR = os.path.join(OUT_BASE_DIR)
while os.path.exists(OUT_DIR):
    out_folder = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = 8))
    OUT_DIR = os.path.join(OUT_BASE_DIR, out_folder)

os.mkdir(OUT_DIR)
print('Saving model to %s' % OUT_DIR)

options = {
    'clip_length': CLIP_LENGTH,
    'batch_size': batch_size,
    'epochs': epochs,
    'latent_dim': latent_dim,
    'num_samples': num_samples,
    'lemma_file': lemma_file,
}
json.dump(options,
    open(os.path.join(OUT_DIR, 'options.json'), 'w'))

with open(lemma_file, 'r') as f:
    for i, line in enumerate(f):
        target_text = line.strip().split()[-1].replace('\u200d', '')
        input_text = line.strip().split()[-2].replace('\u200d', '')
        
        if len(input_text) > CLIP_LENGTH:
            target_text = target_text[-CLIP_LENGTH:]
            input_text = input_text[-CLIP_LENGTH:]
        input_texts.append(input_text)
        #  We use "tab" as the "start sequence" character
        #  for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        target_texts.append(target_text)

        for c in input_text:
            input_characters.add(c)
        for c in target_text:
            target_characters.add(c)

input_texts = input_texts[:num_samples]
target_texts = target_texts[:num_samples]

# Add Devanagari unicode
devanagari = list(range(ord('\u0900'), ord('\u0950'))) + list(range(ord('\u0968'), ord('\u0970')))
devanagari = [chr(i) for i in devanagari]
devanagari = set(devanagari)
input_characters = input_characters.union(devanagari)
target_characters = input_characters.union(devanagari)

# Add <PAD>=' ', <START>='\t', <END>='\n',
# to target_char
target_characters.add(' ')
target_characters.add('\t')
target_characters.add('\n')

# Add <PAD>=' ' to input_char
input_characters.add(' ')

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
# Dump dictionary to file
json.dump([input_token_index, target_token_index], 
    open(os.path.join(OUT_DIR, 'dict.json'), 'w'))

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))  # None is the unknown number of time-series steps
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
csvlogger = CSVLogger(os.path.join(OUT_DIR, 'training.log'), append=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[csvlogger])

model.save(os.path.join(OUT_DIR, 'model.h5'))

print("Outputs saved in %s" % OUT_DIR)

# Inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

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
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        output_char = sampled_char
        decoded_sentence += output_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
