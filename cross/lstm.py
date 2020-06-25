import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import sys

from module import Encoder, Decoder, TransformerEncoder, Embedding, Dense
from module import ReduceLRonPlateau, EarlyStopping, swish
from train_module import TrainStep, Run
from helper import *

parser = argparse.ArgumentParser(description="Train character seq2seq model", 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num-samples", dest="num_samples", required=False,
                    help="max number of samples for training", default=None,
                    type=int)
parser.add_argument("--units", dest="units", required=False,
                    help="size of LSTM unit", default=100,
                    type=int)
parser.add_argument("--embed-dim", dest="embed_dim", required=False,
                    help="size of embedding vector", default=32,
                    type=int)
parser.add_argument("--epochs", dest="epochs", required=False,
                    help="number of epochs for each phase", default='20,20,20,20',
                    type=str)
parser.add_argument("--batch-size", dest="batch_size", required=False,
                    help="Batch size", default='16,16,4,4',
                    type=str)
parser.add_argument("--lr", dest="lr", required=False,
                    help="Initial learning rate", default='1e-3,1e-3,5e-4,5e-4',
                    type=str)
parser.add_argument("--dropout", dest="dropout", required=False,
                    help="Dropout rate", default=0.2,
                    type=float)
parser.add_argument("--L1", dest="L1", required=False,
                    help="High resource language directory", default=None,
                    type=str)
parser.add_argument("--L2", dest="L2", required=True,
                    help="Low resource language directory", type=str)
parser.add_argument("--out-dir", dest="out_dir", required=True,
                    help="output directory", type=str)
parser.add_argument("--mask", dest="mask", required=False,
                    help="masking level: 0 = no mask, 1 = mask to attention",
                    default=0, type=int)
parser.add_argument("--load", dest="load", required=False,
                    help="Which phases to pre-load",
                    default='0,0,0,0', type=str)
parser.add_argument("--skip", dest="skip", required=False,
                    help="Which phases to avoid training",
                    default='0,0,0,0', type=str)
parser.add_argument("--load-dir", dest="load_dir", required=False,
                    help="Directory to load from", default=None, type=str)
parser.add_argument("--dev-size", dest="dev_size", required=False,
                    help="Size of validation/dev set", default=None, type=int)
args = parser.parse_args()

# Start and End tokens - check in helper.py too
START_TOK, END_TOK = '<', '>'

num_samples = args.num_samples

embedding_dim = args.embed_dim
units = args.units
mask = args.mask

batch_size = list(map(int, args.batch_size.split(',')))
lr = list(map(float, args.lr.split(',')))
epochs = list(map(int, args.epochs.split(',')))
load = list(map(bool, map(int, args.load.split(','))))
skip = list(map(bool, map(int, args.skip.split(','))))
load_dir = args.load_dir

num_phases = 4

# Check validity of input arguments
assert len(batch_size) == num_phases
assert len(lr) == num_phases
assert len(epochs) == num_phases
assert len(load) == num_phases
assert len(skip) == num_phases
if any(load):
    assert load_dir is not None

# Create new output directory
OUT_BASE_DIR = args.out_dir
out_folder = ''
while os.path.exists(os.path.join(OUT_BASE_DIR, out_folder)):
    out_folder = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k = 8))
OUT_DIR = os.path.join(OUT_BASE_DIR, out_folder)
os.makedirs(OUT_DIR, exist_ok=True)
print('Files and logs saved to %s' % OUT_DIR)

# Set up logger
logger = logging.getLogger('training')
hdlr = logging.FileHandler(os.path.join(OUT_DIR, 'training.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# Load data from files to variables
# Each tensor is a tuple: (input, target, tag)
train_tensors_L1, test_tensors_L1, val_tensors_L1, tokenizer_L1 = load_ttd_files(os.path.join(args.L1, 'high'))
train_tensors_L2, test_tensors_L2, val_tensors_L2, tokenizer_L2 = load_ttd_files(os.path.join(args.L2, 'low'))

# Max length is a tuple: (input, target, tag)
max_length_L1 = (max(train_tensors_L1[0].shape[1], val_tensors_L1[0].shape[1]), \
                 max(train_tensors_L1[1].shape[1], val_tensors_L1[1].shape[1]), \
                 max(train_tensors_L1[2].shape[1], val_tensors_L1[2].shape[1]))
max_length_L2 = (max(train_tensors_L2[0].shape[1], val_tensors_L2[0].shape[1]), \
                 max(train_tensors_L2[1].shape[1], val_tensors_L2[1].shape[1]), \
                 max(train_tensors_L2[2].shape[1], val_tensors_L2[2].shape[1]))

## Show length
print('L1', len(train_tensors_L1[0]), len(train_tensors_L1[1]), 
    len(test_tensors_L1[0]), len(test_tensors_L1[1]))
print('L2', len(train_tensors_L2[0]), len(train_tensors_L2[1]), 
    len(val_tensors_L1[0]), len(val_tensors_L1[1]))
logger.debug('L1 training (input, target) tensor %d %d' % (
    len(train_tensors_L1[0]), len(train_tensors_L1[1])))
logger.debug('L1 validating (input, target) tensor %d %d' % (
    len(val_tensors_L1[0]), len(val_tensors_L1[1])))
logger.debug('L2 training (input, target) tensor %d %d' % (
    len(train_tensors_L2[0]), len(train_tensors_L2[1])))
logger.debug('L2 validating (input, target) tensor %d %d' % (
    len(val_tensors_L2[0]), len(val_tensors_L2[1])))

vocab_size_L1, vocab_tag_size_L1 = len(tokenizer_L1[0].word_index)+2, len(tokenizer_L1[1].word_index)+2
vocab_size_L2, vocab_tag_size_L2 = len(tokenizer_L2[0].word_index)+2, len(tokenizer_L2[1].word_index)+2

logger.debug('Units %d' % units)
logger.debug('Embedding dim %d' % embedding_dim)
logger.debug('L1 Vocabulary size %d' % (vocab_size_L1))
logger.debug('L1 Tag size %d' % (vocab_tag_size_L1))
logger.debug('L2 Vocabulary size %d' % (vocab_size_L2))
logger.debug('L2 Tag size %d' % (vocab_tag_size_L2))

## Save tokenizers
save_tokenizer(tokenizer_L1, os.path.join(OUT_DIR, 'tokenizer_L1'))
save_tokenizer(tokenizer_L2, os.path.join(OUT_DIR, 'tokenizer_L2'))

# Create copy phase datasets
copy_tag = tokenizer_L1[1].texts_to_sequences([['COPY']])
copy_train_tensors_L1 = create_copy_dataset_from_tensors(*train_tensors_L1, copy_tag=copy_tag)
copy_val_tensors_L1 = create_copy_dataset_from_tensors(*val_tensors_L1, copy_tag=copy_tag)

copy_tag = tokenizer_L2[1].texts_to_sequences([['COPY']])
copy_train_tensors_L2 = create_copy_dataset_from_tensors(*train_tensors_L2, copy_tag=copy_tag)
copy_val_tensors_L2 = create_copy_dataset_from_tensors(*val_tensors_L2, copy_tag=copy_tag)

copy_train_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                            copy_train_tensors_L1).shuffle(len(copy_train_tensors_L1[0]))
copy_val_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                            copy_val_tensors_L1).shuffle(len(copy_val_tensors_L1[0]))
copy_train_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                            copy_train_tensors_L2).shuffle(len(copy_train_tensors_L2[0]))
copy_val_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                            copy_val_tensors_L2).shuffle(len(copy_val_tensors_L2[0]))

if args.dev_size:
    copy_val_dataset_L1 = copy_val_dataset_L1.take(args.dev_size)
    copy_val_dataset_L2 = copy_val_dataset_L2.take(args.dev_size)

# Create main phase datasets
train_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                    train_tensors_L1).shuffle(len(train_tensors_L1[0]))
val_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                    val_tensors_L1).shuffle(len(val_tensors_L1[0]))
train_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                    train_tensors_L2).shuffle(len(train_tensors_L2[0]))
val_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                    val_tensors_L2).shuffle(len(val_tensors_L2[0]))

if args.dev_size:
    val_dataset_L1 = val_dataset_L1.take(args.dev_size)
    val_dataset_L2 = val_dataset_L2.take(args.dev_size)

# Create the modules of the model
char_embedding_L1 = Embedding(vocab_size_L1, embedding_dim)
char_embedding_L2 = Embedding(vocab_size_L2, embedding_dim)

tag_embedding_L1 = Embedding(vocab_tag_size_L1, embedding_dim)
tag_embedding_L2 = Embedding(vocab_tag_size_L2, embedding_dim)

char_encoder = Encoder(units, rate=args.dropout)
tag_encoder = TransformerEncoder(num_layers=1, d_model=embedding_dim,
                                 num_heads=1, dff=units, rate=args.dropout)
decoder = Decoder(units, rate=args.dropout)

fc_L1 = Dense(vocab_size_L1, activation=swish)
fc_L2 = Dense(vocab_size_L2, activation=swish)

# Set up optimizers for phases
optimizer_P1 = tf.keras.optimizers.Adam(lr[0])
optimizer_P2 = tf.keras.optimizers.Adam(lr[1])
optimizer_P3 = tf.keras.optimizers.Adam(lr[2])
optimizer_P4 = tf.keras.optimizers.Adam(lr[3])

# Save all settings
options = {
    'num_samples': num_samples,
    'epochs': epochs,
    'units': units,
    'embedding': embedding_dim,
    'batch_size': batch_size,
    'lr': lr,
    'dropout': args.dropout,
    'L1': args.L1,
    'L2': args.L2,
    'out_dir': OUT_DIR,
    'mask': mask,
    'tag_encoder': {
        'num_layers': 1, 
        'd_model': units, 
        'num_heads': 1,
        'dff': embedding_dim
    },
    'dev_size': args.dev_size,
    'skip': skip,
    'load': load,
    'load-dir': load_dir,
    'source': sys.argv[0]
}
json.dump(options, open(os.path.join(OUT_DIR, 'options.json'), 'w'))

train_step = TrainStep(char_encoder, tag_encoder, decoder)
run = Run(train_step, logger)
# @tf.function

def plot_wrapper(inp, out, plots, directory, fname, tokenizer):
    # inp is a batch_size 1 dataset
    # out is a batch_size 1 output
    char_plot, tag_plot = plots[0][0], plots[1][0]
    char_input, tag_input, output = inp[0].numpy(), inp[1].numpy(), out.numpy()

    char_input = tokenizer[0].sequences_to_texts(char_input)
    tag_input = tokenizer[1].sequences_to_texts(tag_input)
    output = tokenizer[0].sequences_to_texts(output)

    char_input = char_input[0][::2]
    tag_input = tag_input[0].split()
    output = output[0][:output[0].find('>'):2]

    plot_attention(char_plot[:len(output), :len(char_input)],
                   [x for x in char_input], [x for x in output],
                   os.path.join(directory, fname + '-enc.png'))
    plot_attention(tag_plot[:len(output), :len(tag_input)],
                   [x for x in tag_input], [x for x in output],
                   os.path.join(directory, fname + '-tag.png'))

def test_dump(fname):
    test_dataset_L2 = tf.data.Dataset.from_tensor_slices(test_tensors_L2)

    ckpt_dict = {
        'step': tf.Variable(0),
        'optimizer': optimizer_P4,
        'char_encoder': char_encoder,
        'tag_encoder': tag_encoder,
        'decoder': decoder,
        'char_embedding': char_embedding_L2,
        'tag_embedding': tag_embedding_L2,
        'fc': fc_L2
    }
    start_token = tokenizer[0].word_index['<']

    train_step.reset()

    with open(os.path.join(OUT_DIR, '%s.csv' % (fname)), 'w') as o:
        total_accuracy = 0
        outputs = [[0 for _ in range(test_tensors_L2[1].shape[1])]]
        for (batch, batch_dataset) in enumerate(test_dataset_L2.batch(10, drop_remainder=True)):
            _, accuracy, _outputs = train_step(batch_dataset,
                                       ckpt_dict['char_embedding'],
                                       ckpt_dict['tag_embedding'],
                                       ckpt_dict['fc'],
                                       start_token,
                                       ckpt_dict['optimizer'],
                                       mode=mode,
                                       training=False,
                                       mask=mask,
                                       return_outputs=True)
            total_accuracy += accuracy
            outputs = np.concatenate([outputs, _outputs.numpy()], axis=0)

        outputs = tokenizer_L2[0].sequences_to_texts(outputs)
        
        for query, expected, output, tag in zip(
                    test_tensors_L2[0], test_tensors_L2[1], outputs[1:], test_tensors_L2[2]):
            
            query = tokenizer_L2[0].sequences_to_texts([query])[0][::2]
            expected = tokenizer_L2[0].sequences_to_texts([expected])[0][::2]
            tag = tokenizer_L2[1].sequences_to_texts([tag])[0]
            output = output[:output.find(END_TOK):2]
            
            print(query, expected, output, tag, sep='\t', file=o)
        print(total_accuracy.numpy(), 16*(batch+1), file=o)

##################################################################
# Phase 1
##################################################################
mode = 'P1'
val_dataset = copy_val_dataset_L1
train_dataset = copy_train_dataset_L1
tokenizer = tokenizer_L1
_batch_size = batch_size[0]
_load = load[0]
_epochs = epochs[0] if not skip[0] else 0

train_step.reset()

# Set up checkpoints for phase 1
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer_P1,
    'char_encoder': char_encoder,
    'tag_encoder': tag_encoder,
    'decoder': decoder,
    'char_embedding': char_embedding_L1,
    'tag_embedding': tag_embedding_L1,
    'fc': fc_L1
}

checkpoint = tf.train.Checkpoint(**ckpt_dict)
checkpoint_manager = create_checkpoint_manager(checkpoint, os.path.join(OUT_DIR, 'ckpt_%s' % (mode)))

metrics = [(-1, 0, 0, 0)]
sample = next(iter(val_dataset.take(1).batch(1)))
start_token = tokenizer[0].word_index['<']
os.mkdir(os.path.join(OUT_DIR, mode))

# reduceLR = ReduceLRonPlateau(ckpt_dict['optimizer'], patience=5, cooldown=10)
earlyStop = EarlyStopping(patience=3, min_delta=0.)

if _load:
    ckpt_dir = os.path.join(load_dir, 'ckpt_%s' % (mode), 'latest')
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print('{}: Restored from {}'.format(mode, latest))
        logger.info('{}: Restored from {}'.format(mode, latest))
        checkpoint.restore(latest)

# Run phase 1
logger.info('Phase %s' % (mode))
for epoch in range(_epochs):
    start = time.time()

    loss, val_loss, val_accuracy = run(train_dataset,
                                       val_dataset,
                                       ckpt_dict['char_embedding'],
                                       ckpt_dict['tag_embedding'],
                                       ckpt_dict['fc'],
                                       start_token,
                                       ckpt_dict['optimizer'],
                                       mode=mode,
                                       mask=mask,
                                       batch_size=_batch_size)

    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        checkpoint_manager['latest'].save()

    # Dump attention plots
    if epoch % 10 == 0:
        random_sample = next(iter(val_dataset.take(1).batch(1)))
        _, _, output, plots = train_step(
                                   random_sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(random_sample, output, plots, os.path.join(OUT_DIR, mode), 'r'+str(epoch), tokenizer)

        _, _, output, plots = train_step(
                                   sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(sample, output, plots, os.path.join(OUT_DIR, mode), str(epoch), tokenizer)

    print('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))
    logger.info('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation Accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))

    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    metrics.append((epoch, loss, val_loss, val_accuracy))
    
    if val_accuracy > metrics[-2][3]:
        checkpoint_manager['accuracy'].save()

    # if reduceLR(metrics[-1][1]):
    #     logger.info('Learning rate now {}'.format(reduceLR.get_lr()))
    if earlyStop(metrics[-1][1]):
        print('Early stopping callback. Breaking training')
        logger.info('Main phase early stopping')
        break

checkpoint_manager['latest'].save()
np.savetxt(os.path.join(OUT_DIR, "loss_%s.csv" % (mode)), metrics, 
            delimiter=",", header='epochs,loss,val_loss,accuracy')

#######         #######
if checkpoint_manager['validation'].latest_checkpoint:
    checkpoint.restore(checkpoint_manager['validation'].latest_checkpoint)
#######         #######

##################################################################
# Phase 2
##################################################################
mode = 'P2'
val_dataset = copy_val_dataset_L2
train_dataset = copy_train_dataset_L2
tokenizer = tokenizer_L2
_batch_size = batch_size[1]
_load = load[1]
_epochs = epochs[1] if not skip[1] else 0

train_step.reset()

# Set up checkpoints for phase 2
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer_P2,
    'char_encoder': char_encoder,
    'tag_encoder': tag_encoder,
    'decoder': decoder,
    'char_embedding': char_embedding_L2,
    'tag_embedding': tag_embedding_L2,
    'fc': fc_L2
}

checkpoint = tf.train.Checkpoint(**ckpt_dict)
checkpoint_manager = create_checkpoint_manager(checkpoint, os.path.join(OUT_DIR, 'ckpt_%s' % (mode)))

metrics = [(-1, 0, 0, 0)]
sample = next(iter(val_dataset.take(1).batch(1)))
start_token = tokenizer[0].word_index['<']
os.mkdir(os.path.join(OUT_DIR, mode))

# reduceLR = ReduceLRonPlateau(ckpt_dict['optimizer'], patience=4, cooldown=10)
earlyStop = EarlyStopping(patience=5, min_delta=0.)

if _load:
    ckpt_dir = os.path.join(load_dir, 'ckpt_%s' % (mode), 'latest')
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print('{}: Restored from {}'.format(mode, latest))
        logger.info('{}: Restored from {}'.format(mode, latest))
        checkpoint.restore(latest)

# Run phase 2
logger.info('Phase %s' % (mode))
for epoch in range(_epochs):
    start = time.time()

    loss, val_loss, val_accuracy = run(train_dataset,
                                       val_dataset,
                                       ckpt_dict['char_embedding'],
                                       ckpt_dict['tag_embedding'],
                                       ckpt_dict['fc'],
                                       start_token,
                                       ckpt_dict['optimizer'],
                                       mode=mode,
                                       mask=mask,
                                       batch_size=_batch_size)

    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        checkpoint_manager['latest'].save()

    # Dump attention plots
    if epoch % 10 == 0:
        random_sample = next(iter(val_dataset.take(1).batch(1)))
        _, _, output, plots = train_step(
                                   random_sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(random_sample, output, plots, os.path.join(OUT_DIR, mode), 'r'+str(epoch), tokenizer)

        _, _, output, plots = train_step(
                                   sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(sample, output, plots, os.path.join(OUT_DIR, mode), str(epoch), tokenizer)

    print('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))
    logger.info('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation Accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))

    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    metrics.append((epoch, loss, val_loss, val_accuracy))
    
    if val_accuracy > metrics[-2][3]:
        checkpoint_manager['accuracy'].save()

    # if reduceLR(metrics[-1][1]):
    #     logger.info('Learning rate now {}'.format(reduceLR.get_lr()))
    if earlyStop(metrics[-1][1]):
        print('Early stopping callback. Breaking training')
        logger.info('Main phase early stopping')
        break

checkpoint_manager['latest'].save()
np.savetxt(os.path.join(OUT_DIR, "loss_%s.csv" % (mode)), metrics, 
            delimiter=",", header='epochs,loss,val_loss,accuracy')

#######         #######
if checkpoint_manager['validation'].latest_checkpoint:
    checkpoint.restore(checkpoint_manager['validation'].latest_checkpoint)
test_dump(mode)
#######         #######

##################################################################
# Phase 3
##################################################################
mode = 'P3'
val_dataset = val_dataset_L1
train_dataset = train_dataset_L1
tokenizer = tokenizer_L1
_batch_size = batch_size[2]
_load = load[2]
_epochs = epochs[2] if not skip[2] else 0

train_step.reset()

# Set up checkpoints for phase 3
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer_P3,
    'char_encoder': char_encoder,
    'tag_encoder': tag_encoder,
    'decoder': decoder,
    'char_embedding': char_embedding_L1,
    'tag_embedding': tag_embedding_L1,
    'fc': fc_L1
}

checkpoint = tf.train.Checkpoint(**ckpt_dict)
checkpoint_manager = create_checkpoint_manager(checkpoint, os.path.join(OUT_DIR, 'ckpt_%s' % (mode)))

metrics = [(-1, 0, 0, 0)]
sample = next(iter(val_dataset.take(1).batch(1)))
start_token = tokenizer[0].word_index['<']
os.mkdir(os.path.join(OUT_DIR, mode))

reduceLR = ReduceLRonPlateau(ckpt_dict['optimizer'], patience=3, cooldown=6)
earlyStop = EarlyStopping(patience=10, min_delta=0.)

if _load:
    ckpt_dir = os.path.join(load_dir, 'ckpt_%s' % (mode), 'validation')
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print('{}: Restored from {}'.format(mode, latest))
        logger.info('{}: Restored from {}'.format(mode, latest))
        checkpoint.restore(latest)

# Run phase 3
logger.info('Phase %s' % (mode))
for epoch in range(_epochs):
    start = time.time()

    loss, val_loss, val_accuracy = run(train_dataset,
                                       val_dataset,
                                       ckpt_dict['char_embedding'],
                                       ckpt_dict['tag_embedding'],
                                       ckpt_dict['fc'],
                                       start_token,
                                       ckpt_dict['optimizer'],
                                       mode=mode,
                                       mask=mask,
                                       batch_size=_batch_size)

    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        checkpoint_manager['latest'].save()

    # Dump attention plots
    if epoch % 5 == 0:
        random_sample = next(iter(val_dataset.take(1).batch(1)))
        _, _, output, plots = train_step(
                                   random_sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(random_sample, output, plots, os.path.join(OUT_DIR, mode), 'r'+str(epoch), tokenizer)

        _, _, output, plots = train_step(
                                   sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(sample, output, plots, os.path.join(OUT_DIR, mode), str(epoch), tokenizer)

    print('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))
    logger.info('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation Accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))

    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    metrics.append((epoch, loss, val_loss, val_accuracy))
    
    if val_accuracy > metrics[-2][3]:
        checkpoint_manager['accuracy'].save()

    if reduceLR(metrics[-1][1]):
        logger.info('Learning rate now {}'.format(reduceLR.get_lr()))
    if earlyStop(metrics[-1][1]):
        print('Early stopping callback. Breaking training')
        logger.info('Main phase early stopping')
        break

checkpoint_manager['latest'].save()
np.savetxt(os.path.join(OUT_DIR, "loss_%s.csv" % (mode)), metrics, 
            delimiter=",", header='epochs,loss,val_loss,accuracy')

#######         #######
if checkpoint_manager['validation'].latest_checkpoint:
    checkpoint.restore(checkpoint_manager['validation'].latest_checkpoint)
test_dump(mode)
#######         #######

##################################################################
# Phase 4
##################################################################
mode = 'P4'
val_dataset = val_dataset_L2
train_dataset = train_dataset_L2
tokenizer = tokenizer_L2
_batch_size = batch_size[3]
_load = load[3]
_epochs = epochs[3] if not skip[3] else 0

train_step.reset()

# Set up checkpoints for phase 4
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer_P4,
    'char_encoder': char_encoder,
    'tag_encoder': tag_encoder,
    'decoder': decoder,
    'char_embedding': char_embedding_L2,
    'tag_embedding': tag_embedding_L2,
    'fc': fc_L2
}

checkpoint = tf.train.Checkpoint(**ckpt_dict)
checkpoint_manager = create_checkpoint_manager(checkpoint, os.path.join(OUT_DIR, 'ckpt_%s' % (mode)))

metrics = [(-1, 0, 0, 0)]
sample = next(iter(val_dataset.take(1).batch(1)))
start_token = tokenizer[0].word_index['<']
os.mkdir(os.path.join(OUT_DIR, mode))

reduceLR = ReduceLRonPlateau(ckpt_dict['optimizer'], patience=3, cooldown=6)
earlyStop = EarlyStopping(patience=10, min_delta=0.)

if _load:
    ckpt_dir = os.path.join(load_dir, 'ckpt_%s' % (mode), 'validation')
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print('{}: Restored from {}'.format(mode, latest))
        logger.info('{}: Restored from {}'.format(mode, latest))
        checkpoint.restore(latest)

# Run phase 4
logger.info('Phase %s' % (mode))
for epoch in range(_epochs):
    start = time.time()

    loss, val_loss, val_accuracy = run(train_dataset,
                                       val_dataset,
                                       ckpt_dict['char_embedding'],
                                       ckpt_dict['tag_embedding'],
                                       ckpt_dict['fc'],
                                       start_token,
                                       ckpt_dict['optimizer'],
                                       mode=mode,
                                       mask=mask,
                                       batch_size=_batch_size)

    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        checkpoint_manager['latest'].save()

    # Dump attention plots
    if epoch % 5 == 0:
        random_sample = next(iter(val_dataset.take(1).batch(1)))
        _, _, output, plots = train_step(
                                   random_sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(random_sample, output, plots, os.path.join(OUT_DIR, mode), 'r'+str(epoch), tokenizer)

        _, _, output, plots = train_step(
                                   sample,
                                   ckpt_dict['char_embedding'],
                                   ckpt_dict['tag_embedding'],
                                   ckpt_dict['fc'],
                                   start_token,
                                   ckpt_dict['optimizer'],
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)

        plot_wrapper(sample, output, plots, os.path.join(OUT_DIR, mode), str(epoch), tokenizer)

    print('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))
    logger.info('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation Accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))

    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    metrics.append((epoch, loss, val_loss, val_accuracy))
    
    if val_accuracy > metrics[-2][3]:
        checkpoint_manager['accuracy'].save()

    if reduceLR(metrics[-1][1]):
        logger.info('Learning rate now {}'.format(reduceLR.get_lr()))
    if earlyStop(metrics[-1][1]):
        print('Early stopping callback. Breaking training')
        logger.info('Main phase early stopping')
        break

checkpoint_manager['latest'].save()
np.savetxt(os.path.join(OUT_DIR, "loss_%s.csv" % (mode)), metrics, 
            delimiter=",", header='epochs,loss,val_loss,accuracy')

#######         #######
if checkpoint_manager['validation'].latest_checkpoint:
    checkpoint.restore(checkpoint_manager['validation'].latest_checkpoint)
test_dump(mode)
#######         #######