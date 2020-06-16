import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import time
import random
import string
import json
import logging

from att_module import Encoder, BahdanauAttention, Decoder
from helper import *

parser = argparse.ArgumentParser(description="Train character seq2seq model", 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num-samples", dest="num_samples", required=False,
                    help="max number of samples for training", default=10000,
                    type=int)
parser.add_argument("--epochs", dest="epochs", required=False,
                    help="number of epochs", default=100,
                    type=int)
parser.add_argument("--latent-dim", dest="latent_dim", required=False,
                    help="size of LSTM unit", default=100,
                    type=int)
parser.add_argument("--embed-dim", dest="embed_dim", required=False,
                    help="size of embedding vector", default=100,
                    type=int)
parser.add_argument("--batch-size", dest="batch_size", required=False,
                    help="Batch size", default=10,
                    type=int)
parser.add_argument("--clip-length", dest="clip_length", required=False,
                    help="Clip length", default=None,
                    type=int)
parser.add_argument("--out-dir", dest="out_dir", required=False,
                    help="Output directory", default='../att_model',
                    type=str)
parser.add_argument("--inp-dir", dest="inp_dir", required=False,
                    help="Input directory", default='../data',
                    type=str)
args = parser.parse_args()

# Create new output directory
OUT_BASE_DIR = args.out_dir
out_folder = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = 8))
OUT_DIR = os.path.join(OUT_BASE_DIR, out_folder)
os.makedirs(OUT_DIR)

# Set up logger
logger = logging.getLogger('training')
hdlr = logging.FileHandler(os.path.join(OUT_DIR, 'training.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# Start and End tokens - check in helper.py too
START_TOK, END_TOK = '<', '>'

DATA_DIR = args.inp_dir
num_samples = args.num_samples
clip_length = args.clip_length
input_file = os.path.join(DATA_DIR, 'train.csv')
dev_file = os.path.join(DATA_DIR, 'dev.csv')

if os.path.exists(dev_file):
    input_tensor_train, target_tensor_train, lang = load_dataset(input_file, num_samples, clip_length)
    input_tensor_val, target_tensor_val, _ = load_dataset(dev_file, num_samples, clip_length, tokenizer=lang)
    max_length_targ, max_length_inp = max(target_tensor_val.shape[1], target_tensor_train.shape[1]), max(input_tensor_train.shape[1], input_tensor_val.shape[1])
else:
    input_tensor, target_tensor, lang = load_dataset(input_file, num_samples, clip_length)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), 
    len(input_tensor_val), len(target_tensor_val))
logger.debug('training (input, target) tensor %d %d' % (
    len(input_tensor_train), len(target_tensor_train)))
logger.debug('validating (input, target) tensor %d %d' % (
    len(input_tensor_val), len(target_tensor_val)))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = args.batch_size
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = args.embed_dim
units = args.latent_dim
vocab_inp_size = len(lang.word_index)+1
vocab_tar_size = len(lang.word_index)+1

logger.debug('Vocabulary size %d' % (len(lang.word_index)+1))
logger.debug('Units %d' % units)
logger.debug('Embedding dim %d' % embedding_dim)
logger.debug('Batch size %d' % BATCH_SIZE)

save_tokeniser(lang, os.path.join(OUT_DIR, 'tokenizer'))

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
# example_input_batch, example_target_batch = next(iter(dataset))
# example_input_batch.shape, example_target_batch.shape
# 
# sample_hidden = encoder.initialize_hidden_state()
# sample_output, sample_hidden, _ = encoder(example_input_batch, [sample_hidden, sample_hidden])
# print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
# print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
#                                       sample_hidden, sample_output)
# 
# print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

EPOCHS = args.epochs

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, validation=False):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden, _ = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([lang.word_index[START_TOK]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    if not validation:
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

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

# Set up checkpoints
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint,
     os.path.join(OUT_DIR, 'tf_ckpts'), max_to_keep=3)

# Save all settings
options = {
    'num_samples': num_samples,
    'clip_length': clip_length,
    'epochs': EPOCHS,
    'units': units,
    'embedding': embedding_dim,
    'batch_size': BATCH_SIZE,
    'max_length_inp': max_length_inp,
    'max_length_targ': max_length_targ
}
json.dump(options, open(os.path.join(OUT_DIR, 'options.json'), 'w'))
print('Files and logs saved to %s' % OUT_DIR)

loss, val_loss = [], []

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    
    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0 and epoch > 0:
        manager.save()

    loss.append(total_loss / steps_per_epoch)
    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    
    # Calculate validation loss
    enc_hidden = encoder.initialize_hidden_state()
    val_total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden, validation=True)
        val_total_loss += batch_loss
    
    val_loss.append(val_total_loss / steps_per_epoch)
    print('Epoch {} Loss {:.4f} Validation {:.4f}'.format(epoch + 1, loss[-1], val_loss[-1]))
    logger.info('Epoch {} Loss {:.4f} Validation {:.4f}'.format(epoch + 1,
                                      loss[-1], val_loss[-1]))

with open(os.path.join(OUT_DIR, 'loss.csv'), 'w') as f:
    f.write('epoch\tloss\tval_loss\n')
    for i, (lo, vo) in enumerate(zip(loss, val_loss)):
        f.write('{}\t{}\t{}\n'.format(i+1, lo, vo))

manager.save()

# Dump weights as well
os.makedirs(os.path.join(OUT_DIR, 'encoder'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'decoder'), exist_ok=True)
encoder.save_weights(os.path.join(OUT_DIR, 'encoder', 'enc-wt'))
decoder.save_weights(os.path.join(OUT_DIR, 'decoder', 'dec-wt'))

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

with open(os.path.join(DATA_DIR, 'test.csv'), 'r') as f, open(os.path.join(OUT_DIR, 'test.csv'), 'w') as o:
    corr = 0
    faul = 0
    for i, line in enumerate(f):
        if i >= min(2000, num_samples // 2):
            break
        line = line.strip()
        _, word, lemma = line.split('\t')
        out, inp = evaluate(word)
        out, inp = out[:-1], inp[1:-1]
        if clip_length is None:
            if out == lemma:
                corr += 1
            else:
                faul += 1
        else:
            if out == lemma[-clip_length:]:
                corr += 1
            else:
                faul += 1
        
        o.write('{}\t{}\t{}\t{}\n'.format(word,lemma,out,inp))

    o.write('{} {}'.format(corr, faul))

with open(os.path.join(DATA_DIR, 'train.csv'), 'r') as f, open(os.path.join(OUT_DIR, 'train.csv'), 'w') as o:
    corr = 0
    faul = 0
    for i, line in enumerate(f):
        if i >= min(2000, num_samples // 2):
            break
        line = line.strip()
        _, word, lemma = line.split('\t')
        out, inp = evaluate(word)
        out, inp = out[:-1], inp[1:-1]
        if clip_length is None:
            if out == lemma:
                corr += 1
            else:
                faul += 1
        else:
            if out == lemma[-clip_length:]:
                corr += 1
            else:
                faul += 1
        
        o.write('{}\t{}\t{}\t{}\n'.format(word,lemma,out,inp))

    o.write('{} {}'.format(corr, faul))
