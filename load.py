## https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import h5py
import pandas
from argparse import ArgumentParser
import os
import random
import string
import json

# Parser
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg

parser = ArgumentParser(description="Load character seq2seq model")
parser.add_argument("--model", dest="model", required=True,
                    help="path to saved .h5 model",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("--options", dest="options", required=True,
                    help="path to options.json file",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("--dict", dest="dict", required=True,
                    help="path to dict.json file",
                    type=lambda x: is_valid_file(parser, x))
args = parser.parse_args()


options = json.load(open(args.options, 'r'))
latent_dim = options['latent_dim']
num_samples = options['num_samples']
CLIP_LENGTH = options['clip_length']
lemma_file = options['lemma_file']
UNK = options['unk_tok']

input_texts, target_texts = [], []
input_characters, target_characters = set(), set()

with open(lemma_file, 'r') as f:
    for i, line in enumerate(f):
        if i <= num_samples:
            continue
        target_text = line.strip().split()[-1]
        input_text = line.strip().split()[-2]
        if len(input_text) > CLIP_LENGTH:
            target_text = target_text[-CLIP_LENGTH:]
            input_text = input_text[-CLIP_LENGTH:]
        input_texts.append(input_text)
        #  We use "tab" as the "start sequence" character
        #  for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        target_texts.append(target_text)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Load token dictionary and create a reverse look-up
input_token_index, target_token_index = json.load(open(args.dict, 'r'))
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        if char not in input_characters:
            char = UNK
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.

# Load saved model
model = load_model(args.model)
print(model.summary())

# Load layers from saved model
encoder_input_layer = model.get_layer(index=0)
encoder_inputs = Input(shape=(None,encoder_input_layer.input_shape[-1]))
encoder = model.get_layer(index=2)

decoder_input_layer = model.get_layer(index=1)
decoder_inputs = Input(shape=(None,decoder_input_layer.input_shape[-1]))
decoder_lstm = model.get_layer(index=3)
decoder_dense = model.get_layer(index=4)

# Inference
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
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
        if sampled_char == UNK:
            if len(decoded_sentence) < len(input_seq):
                output_char = np.argmax(input_seq[0,len(decoded_sentence)])
                output_char = reverse_input_char_index[output_char]
            else:
                output_char = decoded_sentence[-1]
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

def input_format(word):
    input_text = word
    input_data = np.zeros(
        (1, len(word)+2, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(input_text):
        if char not in input_characters:
            raise ValueError('Invalid char %s in %s' % (char, input_text))
        input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.

for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
