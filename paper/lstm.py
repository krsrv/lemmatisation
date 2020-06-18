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

from module import Encoder, Decoder, TagEncoder
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
parser.add_argument("--inc-tags", dest="inc_tags", required=False,
                    help="Use tags", action='store_true')
parser.add_argument("--use-att", dest="use_att", required=False,
                    help="Use attention for character encoder",
                    action='store_true')
args = parser.parse_args()

# Create new output directory
OUT_BASE_DIR = out_folder = args.out_dir
while os.path.exists(os.path.join(OUT_BASE_DIR, out_folder)):
    out_folder = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = 8))
OUT_DIR = os.path.join(OUT_BASE_DIR, out_folder)
os.makedirs(OUT_DIR, exist_ok=True)

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
inc_tags = args.inc_tags

# Load data from files to variables
if os.path.exists(dev_file):
    tensor, tokenizer = load_dataset(
        input_file, num_samples, clip_length, inc_tags=inc_tags)
    tensor_val, _ = load_dataset(
        dev_file, num_samples, clip_length, tokenizer=tokenizer, inc_tags=inc_tags)

    input_tensor_train, target_tensor_train, tag_tensor_train = tensor
    input_tensor_val, target_tensor_val, tag_tensor_val = tensor_val

    input_tensor = input_tensor_train + input_tensor_val
    target_tensor = target_tensor_train + target_tensor_val
    tag_tensor = tag_tensor_train + target_tensor_val
else:
    tensor, tokenizer = load_dataset(
        input_file, num_samples, clip_length, inc_tags=inc_tags)

    input_tensor, target_tensor, tag_tensor = tensor[0], tensor[1], tensor[2]
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, \
        target_tensor_val, tag_tensor_train, tag_tensor_val = \
        train_test_split(input_tensor, tensor_train, tag_tensor, test_size=0.2)
    
max_length_targ = max(target_tensor_val.shape[1], target_tensor_train.shape[1])
max_length_inp =  max(input_tensor_train.shape[1], input_tensor_val.shape[1])
max_length_tag = max(tag_tensor_train.shape[1], tag_tensor_val.shape[1])

## Language and tag tokenizers
lang, tag_tokenizer = tokenizer

## Show length
print(len(input_tensor_train), len(target_tensor_train), 
    len(input_tensor_val), len(target_tensor_val))
logger.debug('training (input, target) tensor %d %d' % (
    len(input_tensor_train), len(target_tensor_train)))
logger.debug('validating (input, target) tensor %d %d' % (
    len(input_tensor_val), len(target_tensor_val)))

BUFFER_SIZE = len(input_tensor)
BATCH_SIZE = args.batch_size
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = args.embed_dim
units = args.latent_dim
vocab_inp_size = len(lang.word_index)+1
vocab_tar_size = len(lang.word_index)+1
vocab_tag_size = len(tag_tokenizer.word_index)+1

logger.debug('Units %d' % units)
logger.debug('Embedding dim %d' % embedding_dim)
logger.debug('Batch size %d' % BATCH_SIZE)
logger.debug('Vocabulary size %d' % (len(lang.word_index)+1))
if inc_tags:
    logger.debug('Tag size %d' % (len(tag_tokenizer.word_index)+1))

## Save tokenizers
save_tokeniser(lang, os.path.join(OUT_DIR, 'tokenizer'))
if inc_tags:
    save_tokeniser(tag, os.path.join(OUT_DIR, 'tag_tokenizer'))

# Create datasets
if inc_tags:
    # Create dataset for the warm-up phase
    copy_tag_tensor = [['COPY'] for _ in range(len(input_tensor))]
    copy_tag_tensor = tag_tokenizer.texts_to_sequences(copy_tag_tensor)

    X = input_tensor + target_tensor
    T = copy_tag_tensor + tag_tensor
    Y = input_tensor + target_tensor

    X_train, X_test, T_train, T_test, Y_train, Y_test = \
        train_test_split(X, T, Y, test_split=0.2)
    copy_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, T_train, Y_train)).shuffle(BUFFER_SIZE)
    copy_val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, T_test, Y_test)).shuffle(BUFFER_SIZE)
    
    copy_dataset = copy_dataset.batch(BATCH_SIZE, drop_remainder=True)
    copy_val_dataset = copy_val_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Create dataset for the main phase
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train, tag_tensor_train)).shuffle(BUFFER_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val, tag_tensor_val)).shuffle(BUFFER_SIZE)
else:
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units)
tag_encoder = None

if inc_tags:
    tag_encoder = TagEncoder(num_layers=1, d_model=100, num_heads=100, 
                         dff=100, input_vocab_size=vocab_tag_size)

decoder = Decoder(vocab_tar_size, embedding_dim, units, inc_tags=inc_tags)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

EPOCHS = args.epochs

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

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def warm_up(inp, targ, enc_hidden, training=True, tag_inp=None, inc_tags=False):
    loss = 0
    outputs = tf.expand_dims([1] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        enc_output, enc_hidden, _ = encoder(inp, enc_hidden)
        tag_output = None
        if inc_tags:
            tag_output = tag_encoder(tag_inp, training=True, mask=None)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([lang.word_index[START_TOK]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output, tag_vec=tag_output, inc_tags=inc_tags)

            loss += loss_function(targ[:, t], predictions)

            # using scheduled sampling
            if random.random() > 0.5 and training:
                dec_input = tf.expand_dims(targ[:, t], 1)
            else:
                dec_input = tf.argmax(predictions, axis=-1)
                if not training:
                    mask = tf.math.logical_not(tf.math.equal(targ[:, t], 0))
                    accuracy = (targ[;, t] == dec_input)
                    accuracy = tf.math.logical_or(accuracy, mask)

                    outputs = tf.add(outputs, tf.cast(accuracy, dtype='int32'))
                dec_input = tf.expand_dims(dec_input, 1)

    accuracy = tf.reduce_sum(tf.cast(outputs == targ.shape[1]), dtype='int32')
    batch_loss = (loss / int(targ.shape[1]))

    if training:
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, accuracy

# Warm-up until 75% accuracy:
while inc_tags:
    enc_hidden = encoder.initialize_hidden_state(BATCH_SIZE)
    
    for (batch, (inp, targ, tag)) in enumerate(copy_dataset.take(steps_per_epoch)):
        batch_loss, _ = train_step(inp, targ, enc_hidden, tag_inp=tag, inc_tags=inc_tags)
        total_loss += batch_loss

        # if batch % 100 == 0:
            # logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   # batch,
                                                   # batch_loss.numpy()))   
    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 10 == 0:
        manager.save()

    # Calculate validation accuracy
    accuracy = None
    enc_hidden = encoder.initialize_hidden_state()

    for (batch, (inp, targ, tag)) in enumerate(copy_val_dataset.take(steps_per_epoch)):
        _, acc = train_step(
            inp, targ, enc_hidden, training=False, tag_inp=tag, inc_tags=inc_tags)
        accuracy += acc

    print('Accuracy {}'.format(accuracy.numpy()))
assert 1 == 2
@tf.function
def fine_tune(inp, targ, enc_hidden, validation=False, tag_inp=None, inc_tags=False):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden, _ = encoder(inp, enc_hidden)
        tag_output = None
        if inc_tags:
            tag_output = tag_encoder(tag_inp, training=True, mask=None)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([lang.word_index[START_TOK]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output, tag_vec=tag_output, inc_tags=inc_tags)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    if not validation:
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def evaluate(sentence, tags, attention_output=False):
    if attention_output:
        attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence, clip_length)

    inputs = [lang.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    if inc_tags:
        tag_input = preprocess_tags(tags)
        tag_input = [tag_tokenizer.word_index[i] for i in tag_input.split()]
        tag_input = tf.convert_to_tensor(tag_input)

        tag_output = tag_encoder(tag_input, training=False, mask=None)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden, enc_c = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([lang.word_index[START_TOK]], 0)

    for t in range(max_length_targ):
        if inc_tags:
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out,
                                                             tag_vecs=tag_output,
                                                             inc_tags=inc_tags)
        else:
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

loss, val_loss = [], []

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    if inc_tags:
        for (batch, (inp, targ, tag)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, tag_inp=tag, inc_tags=inc_tags)
            total_loss += batch_loss

            if batch % 100 == 0:
                logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
    else:
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

    if inc_tags:
        for (batch, (inp, targ, tag)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(
                inp, targ, enc_hidden, validation=True, tag_inp=tag, inc_tags=inc_tags)
            val_total_loss += batch_loss
    else:
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(
                inp, targ, enc_hidden, validation=True)
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
        tag, word, lemma = line.split('\t')
        out, inp = evaluate(word, tag)
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
        tag, word, lemma = line.split('\t')
        out, inp = evaluate(word, tag)
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
