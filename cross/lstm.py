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

from module import Encoder, Decoder, TransformerEncoder, Embedding, Dense
from module import create_padding_mask, ReduceLRonPlateau, EarlyStopping
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
                    help="Initial learning rate", default='1e-4,1e-4,5e-5,5e-5',
                    type=str)
parser.add_argument("--dropout", dest="dropout", required=False,
                    help="Dropout rate", default=0.2,
                    type=float)
parser.add_argument("--L1", dest="L1", required=False,
                    help="High resource language directory", default=None,
                    type=str)
parser.add_argument("--L2", dest="L2", required=False,
                    help="Low resource language directory",
                    default=0, type=int)
parser.add_argument("--out-dir", dest="out_dir", required=False,
                    help="output directory",
                    default=0, type=int)
parser.add_argument("--copy-threshold", dest="copy_threshold", required=False,
                    help="Accuracy threshold (0-1) for warmup phase", 
                    default=0.75, type=float)
parser.add_argument("--mask", dest="mask", required=False,
                    help="masking level: 0 = no mask, 1 = mask to attention",
                    default=0, type=int)
args = parser.parse_args()

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

# Start and End tokens - check in helper.py too
START_TOK, END_TOK = '<', '>'

num_samples = args.num_samples

embedding_dim = args.embed_dim
units = args.units
mask = args.mask

batch_size = list(map(int, args.batch_size.split(',')))
lr = list(map(float, args.lr.split(',')))
epochs = list(map(int, args.epochs.split(',')))

# Load data from files to variables
# Each tensor is a tuple: (input, target, tag)
train_tensors_L1, test_tensors_L1, val_tensors_L1, tokenizer_L1 = load_ttd(os.path.join(args.L1, 'high'))
train_tensors_L2, test_tensors_L2, val_tensors_L2, tokenizer_L2 = load_ttd(os.path.join(args.L2, 'low'))

# Max length is a tuple: (input, target, tag)
max_length_L1 = (max(train_tensors_L1[0].shape[1], dev_tensors_L1[0].shape[1]), \
                 max(train_tensors_L1[1].shape[1], dev_tensors_L1[1].shape[1]), \
                 max(train_tensors_L1[2].shape[1], dev_tensors_L1[2].shape[1]))
max_length_L2 = (max(train_tensors_L2[0].shape[1], dev_tensors_L2[0].shape[1]), \
                 max(train_tensors_L2[1].shape[1], dev_tensors_L2[1].shape[1]), \
                 max(train_tensors_L2[2].shape[1], dev_tensors_L2[2].shape[1]))

## Show length
print('L1', len(train_tensors_L1[0]), len(train_tensors_L1[1]), 
    len(test_tensors_L1[0]), len(test_tensors_L1[1]))
print('L2', len(train_tensors_L2[0]), len(train_tensors_L2[1]), 
    len(dev_tensors_L1[0]), len(dev_tensors_L1[1]))
logger.debug('L1 training (input, target) tensor %d %d' % (
    len(train_tensors_L1[0]), len(train_tensors_L1[1])))
logger.debug('L1 validating (input, target) tensor %d %d' % (
    len(dev_tensors_L1[0]), len(dev_tensors_L1[1])))
logger.debug('L2 training (input, target) tensor %d %d' % (
    len(train_tensors_L2[0]), len(train_tensors_L2[1])))
logger.debug('L2 validating (input, target) tensor %d %d' % (
    len(dev_tensors_L2[0]), len(dev_tensors_L2[1])))

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
                            copy_train_tensors_L1).shuffle(copy_train_tensors_L1.shape[0])
copy_val_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                            copy_val_tensors_L1).shuffle(copy_val_tensors_L1.shape[0])
copy_train_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                            copy_train_tensors_L2).shuffle(copy_train_tensors_L2.shape[0])
copy_val_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                            copy_val_tensors_L2).shuffle(copy_val_tensors_L2.shape[0])

# Create main phase datasets
train_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                    train_tensors_L1).shuffle(train_tensors_L1[0].shape[0])
val_dataset_L1 = tf.data.Dataset.from_tensor_slices(
                    val_tensors_L1).shuffle(val_tensors_L1[0].shape[0])
train_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                    train_tensors_L2).shuffle(train_tensors_L2[0].shape[0])
val_dataset_L2 = tf.data.Dataset.from_tensor_slices(
                    val_tensors_L2).shuffle(val_tensors_L2[0].shape[0])

# Create the modules of the model
char_embedding_L1 = Embedding(vocab_size_L1, embedding_dim)
char_embedding_L2 = Embedding(vocab_size_L2, embedding_dim)

char_encoder = Encoder(units, rate=args.dropout)
tag_encoder = TransformerEncoder(num_layers=1, d_model=units, num_heads=1, 
                                 dff=256, rate=args.dropout)
decoder = Decoder(embedding_dim, units, inc_tags=inc_tags, rate=args.dropout)

fc_L1 = Dense(vocab_size_L1, activation='swish')
fc_L2 = Dense(vocab_size_L2, activation='swish')

# Set up optimizers for phases
optimizer_P1 = tf.keras.optimizers.Adam(lr[0])
optimizer_P2 = tf.keras.optimizers.Adam(lr[1])
optimizer_P3 = tf.keras.optimizers.Adam(lr[2])
optimizer_P4 = tf.keras.optimizers.Adam(lr[3])

# Save all settings
options = {
    'num_samples': num_samples,
    'epochs': EPOCHS,
    'units': units,
    'embedding': embedding_dim,
    'batch_size': BATCH_SIZE,
    'lr': args.lr,
    'dropout': args.dropout,
    'L1': args.L1,
    'L2': args.L2
    'out_dir': OUT_DIR,
    'mask': mask,
    'copy_threshold': args.copy_threshold,
    'tag_encoder': {
        'num_layers': 1, 
        'units': units, 
        'num_heads': 1,
        'dff': 256}
}
json.dump(options, open(os.path.join(OUT_DIR, 'options.json'), 'w'))

@tf.function
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(batch_dataset, embedder, dense_fc, start_token, optimizer, mask=0,
               mode='P1', training=True, return_outputs=False, return_attention_plots=False):
    loss = 0
    count = [True for _ in range(inputs.shape[0])]
    if return_outputs and not training:
        outputs = [[] for _ in range(inputs.shape[0])]
    if return_attention_plots and not training:
        char_attention_plot = np.zeros((targets.shape[1], inputs.shape[1]))
        tag_attention_plot = np.zeros((targets.shape[1], tags.shape[1]))

    inputs, targets, tags = batch_dataset

    with tf.GradientTape() as tape:
        embedded_inputs = embedder(inputs)
        enc_output, enc_hidden, enc_c = encoder(embedded_inputs,
                                                training=training)

        tag_mask = create_padding_mask(tag_inp, 'transformer')
        tag_output = tag_encoder(tag_inp, training=training, mask=tag_mask)

        dec_states = (enc_hidden, enc_c)
        dec_inputs = tf.expand_dims([start_token] * BATCH_SIZE, 1)
        
        if mask == 1:
            enc_mask = create_padding_mask(inp, 'structure')
            tag_mask = create_padding_mask(tag_inp, 'luong')
        else:
            enc_mask, tag_mask = None, None
        
        decoder.reset()
        for t in range(1, targ.shape[1]):
            embedded_inputs = embedder(dec_inputs)
            outputs, dec_states, attention_weights = decoder(embedded_inputs,
                                                            dec_states,
                                                            enc_output,
                                                            tag_output,
                                                            enc_mask=enc_mask,
                                                            tag_mask=tag_mask,
                                                            training=training)
            predictions = dense_fc(outputs)

            loss += loss_function(targ[:, t], predictions)

            if training:
                if mode in ['P1', 'P2', 'P3']:
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
                elif mode in ['P4']:
                    # using scheduled sampling
                    if random.random() > 0.5:
                        dec_input = tf.expand_dims(targ[:, t], 1)
                    else:
                        dec_input = tf.argmax(predictions, axis=-1, output_type=tf.int32)
                        dec_input = tf.expand_dims(dec_input, 1)
            else:
                # calculating accuracy when running validation
                dec_input = tf.argmax(predictions, axis=-1, output_type=tf.int32)
                
                mask = tf.math.equal(targ[:, t], 0)
                accuracy = (targ[:, t] == dec_input)
                accuracy = tf.math.logical_or(accuracy, mask)
                count = tf.math.logical_and(count, accuracy)
                
                dec_input = tf.expand_dims(dec_input, 1)
                if return_outputs:
                    outputs = tf.concat([outputs, dec_input], axis=-1)
                if return_attention_plots:
                    buff = tf.reshape(attention_weights[0], (-1, ))
                    char_attention_plot[t] = buff.numpy()

                    buff = tf.reshape(attention_weights[1], (-1, ))
                    tag_attention_plot[t] = buff.numpy()
            
    count = tf.reduce_sum(tf.cast(count, dtype='int32'))

    batch_loss = (loss / int(targ.shape[1]))

    if training:
        if mode == 'P1':
            variables = encoder.trainable_variables + decoder.trainable_variables + \
                        embedder.trainable_variables + dense_fc.trainable_variables
        elif mode == 'P2':
            variables = embedder.trainable_variables + dense_fc.trainable_variables
        elif mode == 'P3':
            variables = encoder.trainable_variables + decoder.trainable_variables + \
                        dense_fc.trainable_variables
        elif mode == 'P4':
            variables = dense_fc.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

    if return_attention_plots and return_outputs:
        return batch_loss, count, outputs, (char_attention_plot, tag_attention_plot)
    elif return_attention_plots:
        return batch_loss, count, (char_attention_plot, tag_attention_plot)
    elif return_outputs:
        return batch_loss, count, outputs
    else:
        return batch_loss, count

@tf.function
def run(train_dataset, val_dataset, embedder, dense_fc, start_token,
        optimizer, mode='P1', mask=0, batch_size=10):
    # Batch the train and val datasets
    train_batch_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_batch_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    total_loss = 0
    
    for (batch, train_batch) in enumerate(train_batch_dataset):
        batch_loss, _ = train_step(train_batch,
                                   embedder,
                                   dense_fc,
                                   start_token,
                                   optimizer,
                                   mask=mask,
                                   mode=mode,
                                   training=True)
        total_loss += batch_loss

        if batch % 100 == 0:
            logger.debug('{} Epoch {} Batch {} Loss {:.4f}'.format(
                                                    mode
                                                    int(checkpoint.step),
                                                    batch,
                                                    batch_loss.numpy()))
    total_loss /= batch
    
    # Calculate validation accuracy
    val_total_loss = 0
    val_total_accuracy = 0

    for (batch, val_batch) in enumerate(val_batch_dataset):
        batch_loss, batch_accuracy = train_step(val_batch,
                                                embedder,
                                                dense_fc,
                                                start_token,
                                                optimizer,
                                                mask=mask,
                                                mode=mode,
                                                training=False)
        val_total_loss += batch_loss
        val_total_accuracy += batch_accuracy
    
    val_total_loss /= batch
    val_total_accuracy /= batch * batch_size
    
    return total_loss, val_total_loss, val_total_accuracy

def plot_wrapper(inp, out, plots, directory, fname, tokenizer):
    inp, out = next(iter(inp)), next(iter(out))

    char_input = tokenizer[0].sequences_to_texts(inp[0][0])
    tag_input = tokenizer[0].sequences_to_texts(inp[1])
    output = tokenizer[0].sequences_to_texts(out)

    char_input = char_input[::2]
    output = output[:output.find('>')]

    plot_attention(plots[0][:len(output), :len(char_input)],
                   [x for x in char_input], [x for x in output],
                   os.path.join(directory, fname + '-enc.png'))
    plot_attention(plots[1][:len(output), :len(tag_input)],
                   [x for x in tag_input], [x for x in output],
                   os.path.join(directory, fname + '-tag.png'))

# Phase 1
# Set up checkpoints for phase 1
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer_P1,
    'char_encoder': char_encoder,
    'tag_encoder': tag_encoder,
    'decoder': decoder,
    'embedding': char_embedding_L1,
    'fc': fc_L1
}

checkpoint = tf.train.Checkpoint(**ckpt_dict)
checkpoint_manager = create_checkpoint_manager(checkpoint, os.path.join(OUT_DIR, 'ckpt_p1'))

metrics = []
sample = copy_val_dataset_L1.take(1)
start_token = tokenizer_L1[0].word_index['<']
mode = 'P1'

# Run phase 1
logger.info('Phase P1')
for epoch in range(20):
    start = time.time()

    loss, val_loss, val_accuracy = run(copy_train_dataset_L1,
                                       copy_val_dataset_L1,
                                       char_embedding_L1,
                                       fc_L1,
                                       start_token,
                                       optimizer_1,
                                       mode=mode,
                                       mask=mask,
                                       batch_size=batch_size[0])
    metrics.append((epoch, loss, val_loss, val_accuracy))

    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        checkpoint_manager['latest'].save()

    # Dump attention plots
    if epoch % 5 == 0:
        random_sample = copy_val_dataset_L1.take(1)
        _, _, _, output, plot = train_step(random_sample.batch(1),
                                   char_embedding_L1,
                                   fc_L1,
                                   start_token,
                                   optimizer_1,
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)
        
        plot_wrapper(random_sample, output, plot, os.path.join(OUT_DIR, mode), 'r'+str(epoch), tokenizer_L1)

        _, _, _, output, plot = train_step(sample.batch(1),
                                   char_embedding_L1,
                                   fc_L1,
                                   start_token,
                                   optimizer_1,
                                   mode=mode,
                                   training=False,
                                   mask=mask,
                                   return_outputs=True,
                                   return_attention_plots=True)
        
        plot_wrapper(sample, output, plot, os.path.join(OUT_DIR, mode), str(epoch), tokenizer_L1)

    print('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))
    logger.info('{}: Epoch {} Loss {:.4f} Validation {:.4f} Validation Accuracy {}'.format(
            mode, epoch+1, loss, val_loss, val_accuracy))

    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    
    if val_accuracy > accuracy[-1]:
        checkpoint_manager['accuracy'].save()

    if total_accuracy >= args.copy_threshold:
        print('Successful. Phase %s over' % (mode))
        logger.info('Phase {} finished with accuracy {}'.format(mode, val_accuracy))
        break

checkpoint_manager['latest'].save()
np.savetxt("loss_%s.csv" % (mode), metrics, delimiter=",", header='epochs,loss,val_loss,accuracy')

# with open(os.path.join(DATA_DIR, 'test.csv'), 'r') as f, \
#      open(os.path.join(OUT_DIR, 'test.csv'), 'w') as o:
#     corr = 0
#     faul = 0
#     if use_ptv:
#         ptvs = load_ptv(os.path.join(DATA_DIR, 'ptv-test-%d.npy' % (ptv_dim)))
#     if args.test_img:
#         os.makedirs(os.path.join(OUT_DIR, 'pictures'))
#     for i, line in enumerate(f):
#         if i >= min(2000, num_samples):
#             break
#         line = line.strip()
#         tag, word, lemma = line.split('\t')
#         ptv = ptvs[i] if use_ptv else None
#         out, inp, *at = evaluate(word, tag, inc_tags=inc_tags, mask=mask_level,
#                             ptv=ptv, cnst_tag=cnst_tag, attention_output=args.test_img)
#         if args.test_img and inc_tags:
#             plot_attention(
#                 at[0][1][:len(out), :len(inp[1].split())+1], 
#                 inp[1].split() + ['blank'], [x for x in out], 
#                 os.path.join(OUT_DIR, 'pictures', str(i) + '-tag.png'))
#         out = out[:-1]
#         if inc_tags:
#             word_inp = inp[0][1:-1]
#             tag_inp = inp[1]
#             output = '{}\t{}\t{}\t{}\t{}'.format(word, lemma, out, word_inp, tag_inp)
#         else:
#             word_inp = inp[1:-1]
#             output = '{}\t{}\t{}\t{}'.format(word, lemma, out, word_inp)
#         if clip_length:
#             lemma_clipped = lemma[-clip_length]
#         else:
#             lemma_clipped = lemma
#         corr += (out == lemma_clipped)
#         faul += (out != lemma_clipped)
#         print(output, file=o)
#     print('{} {}'.format(corr, faul), file=o)
