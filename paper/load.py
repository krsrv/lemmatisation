import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import json
import pickle

from module import Encoder, TagEncoder, Decoder, create_padding_mask
from helper import *

parser = argparse.ArgumentParser(description="Load character seq2seq model", 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir", dest="dir", required=False,
                    help="directory where files are stored", default='.',
                    type=str)
parser.add_argument("--use-weights", dest="use_weights", required=False,
                    help="use weight files if enabled. Otherwise checkpoints are used for loading models",
                    action='store_true')
parser.add_argument("--ckpt-dir", dest="ckpt_dir", required=False,
                    help="directory to load checkpoints from (relative to dir)", default=None,
                    type=str)
parser.add_argument("--test-file", dest="test_file", required=False,
                    help="if supplied, run tests on given file", default='',
                    type=str)
parser.add_argument("--out-file", dest="out_file", required=False,
                    help="output of test file", default=None,
                    type=str)
args = parser.parse_args()

# Start and End tokens - check in helper.py too
START_TOK, END_TOK = '<', '>'

# Load options
options = json.load(open(os.path.join(args.dir, 'options.json'), 'r'))

# Populate variables using options
clip_length = options['clip_length'] \
    if 'clip_length' in options.keys() else None
inc_tags = options['inc_tags'] \
    if 'inc_tags' in options.keys() else False

EPOCHS = options['epochs'] \
    if 'epochs' in options.keys() else 100
BATCH_SIZE = options['batch_size'] \
    if 'batch_size' in options.keys() else 10
embedding_dim = options['embedding']
dropout = options['dropout'] \
    if 'dropout' in options.keys() else 0.2
mask_level = options['mask'] \
    if 'mask' in options.keys() else 0

units = options['units']
num_samples = options['num_samples']

# Load tokenizer
lang = pickle.load(open(os.path.join(args.dir, 'tokenizer'), 'rb'))
if inc_tags:
    tag_tokenizer = pickle.load(open(os.path.join(args.dir, 'tag_tokenizer'), 'rb'))
    tokenizer = (lang, tag_tokenizer)
else:
    tokenizer = lang

# Load vocabulary and sequence length settings
max_length_targ = options['max_length_targ'] \
    if 'max_length_targ' in options.keys() else 15
max_length_inp = options['max_length_inp'] \
    if 'max_length_inp' in options.keys() else 15
if inc_tags:
    max_length_tag = options['max_length_tag'] \
        if 'max_length_tag' in options.keys() else 5

vocab_inp_size = len(lang.word_index)+2
vocab_tar_size = len(lang.word_index)+2
if inc_tags:
    vocab_tag_size = len(tag_tokenizer.word_index)+2

# Load checkpoint directory
if args.ckpt_dir:
    assert os.path.exists(os.path.join(args.dir, args.ckpt_dir))
    ckpt_dir = os.path.join(args.dir, args.ckpt_dir)
elif not args.use_weights:
    if os.path.exists(os.path.join(args.dir, 'tf_ckpts')):
        ckpt_dir = os.path.join(args.dir, 'tf_ckpts')
    elif os.path.exists(os.path.join(args.dir, 'val_ckpt')):
        ckpt_dir = os.path.join(args.dir, 'val_ckpt')
    elif os.path.exists(os.path.join(args.dir, 'acc_ckpt')):
        ckpt_dir = os.path.join(args.dir, 'acc_ckpt')
    elif os.path.exists(os.path.join(args.dir, 'latest_ckpt')):
        ckpt_dir = os.path.join(args.dir, 'latest_ckpt')
    elif os.path.exists(os.path.join(args.dir, 'copy_ckpt')):
        ckpt_dir = os.path.join(args.dir, 'copy_ckpt')

# Helper function to create dataset from filepath
def _create_dataset(path, tokenizer=None, return_tf_dataset=False):
    if tokenizer is None:
        tokenizer = create_tokenizer(path)
    tensor, _ = load_dataset(path, tokenizer=tokenizer)

    input_tensor_train, input_tensor_val, target_tensor_train, \
        target_tensor_val, tag_tensor_train, tag_tensor_val = \
        train_test_split(tensor[0], tensor[1], tensor[2], test_size=0.2)
    
    if return_tf_dataset:
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train, tag_tensor_train))
        dataset = dataset.shuffle(len(input_tensor_train))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset

    return (input_tensor_train, target_tensor_train, tag_tensor_train), \
        (input_tensor_val, target_tensor_val, tag_tensor_val)

def evaluate(sentence, tags, attention_output=False, inc_tags=False, mask=0):
    if attention_output:
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        tag_attention_plot = np.zeros((max_length_targ, max_length_tag))

    sentence = preprocess_sentence(sentence, clip_length)

    inputs = lang.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)
    enc_mask = create_padding_mask(inputs, 'lstm') if mask == 2 else None
    
    enc_out, enc_hidden, enc_c = encoder(inputs, training=False, mask=enc_mask)

    if inc_tags:
        tags = preprocess_tags(tags)
        tag_input = tag_tokenizer.texts_to_sequences([tags])
        tag_input = tf.keras.preprocessing.sequence.pad_sequences(tag_input,
                                                         maxlen=max_length_tag,
                                                         padding='post')
        tag_input = tf.convert_to_tensor(tag_input)

        tag_mask = create_padding_mask(tag_input, 'transformer')
        tag_output = tag_encoder(tag_input, training=False, mask=tag_mask)
    else:
        tag_output, tag_mask = None, None

    result = ''

    dec_states = (enc_hidden, enc_c)
    dec_input = tf.expand_dims([lang.word_index[START_TOK]], 0)
    if mask == 1 or mask == 2:
        enc_mask = create_padding_mask(inputs, 'luong')
        tag_mask = create_padding_mask(tag_input, 'luong') if inc_tags else None
    else:
        enc_mask, tag_mask = None, None

    for t in range(max_length_targ):
        predictions, dec_states, attention_weights = decoder(dec_input,
                                                             dec_states,
                                                             enc_out,
                                                             tag_vecs=tag_output,
                                                             enc_mask=enc_mask, tag_mask=tag_mask,
                                                             training=False)
        
        if attention_output:
            # storing the attention weights to plot later on
            if inc_tags:
                buff = tf.reshape(attention_weights[0], (-1, ))
                attention_plot[t] = buff.numpy()

                buff = tf.reshape(attention_weights[1], (-1, ))
                tag_attention_plot[t] = buff.numpy()
            else:
                buff = tf.reshape(attention_weights[0], (-1, ))
                attention_plot[t] = buff.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += lang.index_word[predicted_id]

        if lang.index_word[predicted_id] == END_TOK:
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    if attention_output:
        if inc_tags:
            return result, (sentence, tags), (attention_plot, tag_attention_plot)
        else:
            return result, sentence, attention_plot
    else:
        if inc_tags:
            return result, (sentence, tags)
        else:
            return result, sentence

encoder = Encoder(vocab_inp_size, embedding_dim, units)
decoder = Decoder(vocab_tar_size, embedding_dim, units, inc_tags=inc_tags)
if inc_tags:
    params = options['tag_encoder']
    tag_encoder = TagEncoder(num_layers = params['num_layers'], 
                             d_model = params['d_model'], 
                             num_heads = params['num_heads'], 
                             dff = params['dff'],
                             input_vocab_size = vocab_tag_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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
latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

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

if os.path.exists(args.test_file):
    out_file = args.out_file if args.out_file else 'out_test.csv'

    with open(os.path.join(args.test_file), 'r') as f, \
         open(os.path.join(args.dir, out_file), 'w') as o:
        corr = 0
        faul = 0
        for i, line in enumerate(f):
            if i >= min(2000, num_samples):
                break
            line = line.strip()
            tag, word, lemma = line.split('\t')
            out, inp = evaluate(word, tag, inc_tags=inc_tags, mask=mask_level)
            out = out[:-1]
            if inc_tags:
                word_inp = inp[0][1:-1]
                tag_inp = inp[1]
                output = '{}\t{}\t{}\t{}\t{}'.format(word, lemma, out, word_inp, tag_inp)
            else:
                word_inp = inp[1:-1]
                output = '{}\t{}\t{}\t{}'.format(word, lemma, out, word_inp)
            if clip_length:
                lemma_clipped = lemma[-clip_length]
            else:
                lemma_clipped = lemma
            corr += (out == lemma_clipped)
            faul += (out != lemma_clipped)
            print(output, file=o)
        print('{} {}'.format(corr, faul), file=o)

