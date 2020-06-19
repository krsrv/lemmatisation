import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import json
import pickle

from att_module import Encoder, BahdanauAttention, Decoder
from helper import *

parser = argparse.ArgumentParser(description="Load character seq2seq model", 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir", dest="dir", required=False,
                    help="directory where files are stored", default='.',
                    type=str)
parser.add_argument("--use-weights", dest="use_weights", required=False,
                    help="use weight files if enabled. Otherwise checkpoints are used for loading models",
                    action='store_true')
parser.add_argument("--test-file", dest="test_file", required=False,
                    help="if supplied, run tests on given file", default=None,
                    type=str)
args = parser.parse_args()

# Start and End tokens - check in helper.py too
START_TOK, END_TOK = '<', '>'

# Load options
options = json.load(open(os.path.join(args.dir, 'options.json'), 'r'))

# Populate variables using options
max_length_targ = options['max_length_targ'] \
    if 'max_length_targ' in options.keys() else 15
max_length_inp = options['max_length_inp'] \
    if 'max_length_inp' in options.keys() else 15
clip_length = options['clip_length'] \
    if 'clip_length' in options.keys() else None
inc_tags = options['inc_tags'] \
    if 'inc_tags' in options.keys else False

EPOCHS = options['epochs'] \
    if 'epochs' in options.keys() else 100
BATCH_SIZE = options['batch_size'] \
    if 'batch_size' in options.keys() else 10
embedding_dim = options['embedding']
units = options['units']
vocab_inp_size = len(lang.word_index)+1
vocab_tar_size = len(lang.word_index)+1
if inc_tags:
    vocab_tag_size = len(tag_tokenizer.word_index)+1

# Load tokenizer
lang = pickle.load(open(os.path.join(args.dir, 'tokenizer'), 'rb'))
if inc_tags:
    tag_tokenizer = pickle.load(open(os.path.join(args.dir, 'tokenizer'), 'rb'))

def _create_dataset(path, return_tf_dataset=False):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w, clip_length) for w in l.split()[1:]]  
        for l in lines[:num_examples]]

    inp_lang, targ_lang = zip(*word_pairs)
    input_tensor, _ = tokenize(inp_lang, lang_tokenizer)
    target_tensor, _ = tokenize(targ_lang, lang_tokenizer)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2)
    
    if return_tf_dataset:
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)).shuffle(len(input_tensor_train))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset

    return input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val

def evaluate(sentence, attention_output=False):
    if attention_output:
        attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence, clip_length)

    inputs = [lang.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden, enc_c = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang.word_index[START_TOK]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        if attention_output:
            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += lang.index_word[predicted_id]

        if lang.index_word[predicted_id] == END_TOK:
            if attention_output:
                return result, sentence, attention_plot
            else:
                return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    if attention_output:
        return result, sentence, attention_plot
    else:
        return result, sentence

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

if inc_tags:
    params = options['tag_encoder']
    tag_encoder = TagEncoder(num_layers = params['num_layers'], 
                             d_model = params['units'], 
                             num_heads = params['num_heads'], 
                             dff = params['dff'],
                             input_vocab_size = vocab_tag_size)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Set up checkpoints
ckpt_dict = {
    'step': tf.Variable(1),
    'optimizer': optimizer,
    'encoder': encoder,
    'decoder': decoder
}
if inc_tags:
    ckpt_dict['tag_encoder'] = tag_encoder

checkpoint = tf.train.Checkpoint(**ckpt_dict)

# manager = tf.train.CheckpointManager(checkpoint, os.path.join(args.dir, 'tf_ckpts'), max_to_keep=3)
latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.dir, 'tf_ckpts'))

if args.use_weights:
    encoder.load_weights(os.path.join(args.dir, 'encoder', 'enc-wt'))
    decoder.load_weights(os.path.join(args.dir, 'decoder', 'dec-wt'))
    print("Restored from {}/(encoder, decoder)".format(args.dir))
else:
    checkpoint.restore(latest_ckpt)
    if latest_ckpt:
        print("Restored from {}".format(latest_ckpt))
    else:
        print("Checkpoint not found. Initializing from scratch.")
