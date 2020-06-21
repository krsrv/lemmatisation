import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

from module import Encoder, Decoder, TagEncoder, create_padding_mask, CustomCallback
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
                    help="size of embedding vector", default=32,
                    type=int)
parser.add_argument("--batch-size", dest="batch_size", required=False,
                    help="Batch size", default=10,
                    type=int)
parser.add_argument("--lr", dest="lr", required=False,
                    help="Initial learning rate", default=0.001,
                    type=float)
parser.add_argument("--dropout", dest="dropout", required=False,
                    help="Dropout rate", default=0.2,
                    type=float)
parser.add_argument("--clip-length", dest="clip_length", required=False,
                    help="Clip length", default=None,
                    type=int)
parser.add_argument("--out-dir", dest="out_dir", required=False,
                    help="Output directory", default='../att_model',
                    type=str)
parser.add_argument("--inp-dir", dest="inp_dir", required=False,
                    help="Input directory", default='../data',
                    type=str)
parser.add_argument("--exc-tags", dest="exc_tags", required=False,
                    help="Use tags", action='store_true')
parser.add_argument("--no-copy", dest="copy", required=False,
                    help="Skip copying/warm-up phase", action='store_false')
args = parser.parse_args()

# Create new output directory
OUT_BASE_DIR = args.out_dir
out_folder = ''
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
inc_tags = not args.exc_tags

# Load data from files to variables
if os.path.exists(dev_file):
    tokenizer = create_tokenizer(input_file, dev_file)
    tensor, tokenizer = load_dataset(
        input_file, num_samples, clip_length, tokenizer=tokenizer)
    tensor_val, _ = load_dataset(
        dev_file, num_samples, clip_length, tokenizer=tokenizer)

    input_tensor_train, target_tensor_train, tag_tensor_train = tensor
    input_tensor_val, target_tensor_val, tag_tensor_val = tensor_val

    # input_tensor = pad(input_tensor_train, input_tensor_val)
    # target_tensor = pad(target_tensor_train, target_tensor_val)
    # tag_tensor = pad(tag_tensor_train, tag_tensor_val)
else:
    tokenizer = create_tokenizer(input_file)
    tensor, tokenizer = load_dataset(
        input_file, num_samples, clip_length, tokenizer=tokenizer)

    input_tensor, target_tensor, tag_tensor = tensor[0], tensor[1], tensor[2]
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, \
        target_tensor_val, tag_tensor_train, tag_tensor_val = \
        train_test_split(input_tensor, target_tensor, tag_tensor, test_size=0.2)
    
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

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = args.batch_size
steps_per_epoch_train = len(input_tensor_train)//BATCH_SIZE
steps_per_epoch_val = len(input_tensor_val)//BATCH_SIZE
embedding_dim = args.embed_dim
units = args.latent_dim
vocab_inp_size = len(lang.word_index)+2
vocab_tar_size = len(lang.word_index)+2
vocab_tag_size = len(tag_tokenizer.word_index)+2

logger.debug('Units %d' % units)
logger.debug('Embedding dim %d' % embedding_dim)
logger.debug('Batch size %d' % BATCH_SIZE)
logger.debug('Vocabulary size %d' % (vocab_inp_size))
if inc_tags:
    logger.debug('Tag size %d' % (vocab_tag_size))

## Save tokenizers
save_tokeniser(lang, os.path.join(OUT_DIR, 'tokenizer'))
if inc_tags:
    save_tokeniser(tag_tokenizer, os.path.join(OUT_DIR, 'tag_tokenizer'))

# Create datasets
# Create dataset for the warm-up phase
copy_tag_tensor = [['COPY']]
copy_tag_tensor = tag_tokenizer.texts_to_sequences(copy_tag_tensor)
copy_tag_tensor = tf.keras.preprocessing.sequence.pad_sequences(
                                copy_tag_tensor,
                                maxlen=max_length_tag,
                                padding='post')
copy_tag_tensor_train = np.repeat(copy_tag_tensor, input_tensor_train.shape[0], axis=0)
copy_tag_tensor_val = np.repeat(copy_tag_tensor, input_tensor_val.shape[0], axis=0)

# The tensors need to be padded to have the same sequence length
X_train = pad(input_tensor_train, target_tensor_train, concatenate=True)
Y_train = X_train[:, :]
T_train = pad(copy_tag_tensor_train, tag_tensor_train)

X_val = pad(input_tensor_val, target_tensor_val, concatenate=True)
Y_val = X_val[:, :]
T_val = pad(copy_tag_tensor_val, tag_tensor_val)

copy_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train, T_train)).shuffle(2*BUFFER_SIZE)
copy_val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, Y_val, T_val)).shuffle(2*BUFFER_SIZE)

copy_dataset = copy_dataset.batch(BATCH_SIZE, drop_remainder=True)
copy_val_dataset = copy_val_dataset.batch(BATCH_SIZE, drop_remainder=True)

# Create dataset for the main phase
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train, tag_tensor_train)).shuffle(BUFFER_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_val, target_tensor_val, tag_tensor_val)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units, rate=args.dropout)
tag_encoder = None

if inc_tags:
    tag_encoder = TagEncoder(num_layers=1, d_model=units, num_heads=1, 
                         dff=256, input_vocab_size=vocab_tag_size,
                         rate=args.dropout)

decoder = Decoder(vocab_tar_size, embedding_dim, units, 
                  inc_tags=inc_tags, rate=args.dropout)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(units)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam(args.lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

EPOCHS = args.epochs

# Set up checkpoints
ckpt_dict = {
    'step': tf.Variable(0),
    'optimizer': optimizer,
    'encoder': encoder,
    'decoder': decoder
}
if inc_tags:
    ckpt_dict['tag_encoder'] = tag_encoder

checkpoint = tf.train.Checkpoint(**ckpt_dict)
copy_manager = tf.train.CheckpointManager(checkpoint,
     os.path.join(OUT_DIR, 'copy_ckpt'), max_to_keep=1)
main_manager = {
    'accuracy': tf.train.CheckpointManager(checkpoint,
        os.path.join(OUT_DIR, 'acc_ckpt'), max_to_keep=1),
    'validation': tf.train.CheckpointManager(checkpoint,
        os.path.join(OUT_DIR, 'val_ckpt'), max_to_keep=1),
    'latest': tf.train.CheckpointManager(checkpoint,
        os.path.join(OUT_DIR, 'latest_ckpt'), max_to_keep=1)
    }

# Save all settings
options = {
    'num_samples': num_samples,
    'clip_length': clip_length,
    'epochs': EPOCHS,
    'units': units,
    'embedding': embedding_dim,
    'batch_size': BATCH_SIZE,
    'max_length_inp': max_length_inp,
    'max_length_targ': max_length_targ,
    'inc_tags': inc_tags,
    'lr': args.lr,
    'dropout': args.dropout,
    'input_dir': DATA_DIR,
    'output_dir': OUT_DIR
}
if inc_tags:
    options['max_length_tag'] = max_length_tag
    options['tag_encoder'] = {
        'num_layers': 1,
        'd_model': units,
        'num_heads': 1,
        'dff': 256,
    }

json.dump(options, open(os.path.join(OUT_DIR, 'options.json'), 'w'))
print('Files and logs saved to %s' % OUT_DIR)

def evaluate(sentence, tags, attention_output=False, inc_tags=False):
    if attention_output:
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        tag_attention_plot = np.zeros((max_length_targ, max_length_tag))

    sentence = preprocess_sentence(sentence, clip_length)

    inputs = lang.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)
    enc_mask = create_padding_mask(inputs)

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

    enc_out, enc_hidden, enc_c = encoder(inputs, training=False)

    dec_states = (enc_hidden, enc_c)
    dec_input = tf.expand_dims([lang.word_index[START_TOK]], 0)

    for t in range(max_length_targ):
        predictions, dec_states, attention_weights = decoder(dec_input,
                                                             dec_states,
                                                             enc_out,
                                                             tag_vecs=tag_output,
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

def output(sentence, fname, tags=None, inc_tags=False):
    ou, ip, at = evaluate(sentence, tags, attention_output=True, inc_tags=inc_tags)
    if inc_tags:
        plot_attention(
            at[0][:len(ou), :len(ip[0])], 
            ip[0], ou, 
            os.path.join(OUT_DIR, fname + '-enc.png'))
        plot_attention(
            at[1][:len(ou), :len(ip[1].split())], 
            ip[1], ou, 
            os.path.join(OUT_DIR, fname + '-tag.png'))
    else:
        plot_attention(
            at[:len(ou), :len(ip)], 
            ip, ou, 
            os.path.join(OUT_DIR, fname + '-enc.png'))
    return ou

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, mode='main', enc_state=None, training=True,
        tag_inp=None, inc_tags=False, return_outputs=False):
    loss = 0
    outputs = [[] for _ in range(BATCH_SIZE)]
    count = [True for _ in range(BATCH_SIZE)]

    with tf.GradientTape() as tape:
        enc_output, enc_hidden, enc_c = encoder(inp, state=enc_state, training=training)

        tag_output, tag_mask = None, None
        if inc_tags:
            tag_mask = create_padding_mask(tag_inp, 'transformer')
            tag_output = tag_encoder(tag_inp, training=True, mask=tag_mask)
            
            # Different mask needed for non-transformer architecture
            tag_mask = None
        
        dec_states = (enc_hidden, enc_c)
        dec_input = tf.expand_dims([lang.word_index[START_TOK]] * BATCH_SIZE, 1)
        
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_states, _ = decoder(dec_input, dec_states, enc_output,
                                                tag_vecs=tag_output,
                                                training=training)

            loss += loss_function(targ[:, t], predictions)

            if training:
                if mode == 'warm-up' or mode == 'main':
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
                elif mode == 'main':
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
            
    count = tf.reduce_sum(tf.cast(count, dtype='int32'))

    batch_loss = (loss / int(targ.shape[1]))

    if training:
        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

    if return_outputs:
        return batch_loss, count, outputs
    else:
        return batch_loss, count

# Warm-up until 75% accuracy:
logger.info('Warm-up phase')
while args.copy:
    start = time.time()
    total_loss = 0
    
    for (batch, (inp, targ, tag)) in enumerate(copy_dataset):
        batch_loss, _ = train_step(inp, targ, mode='warm-up', 
                                   tag_inp=tag, 
                                   inc_tags=inc_tags)
        total_loss += batch_loss

        if batch % 100 == 0:
            logger.info('Warm-up Epoch {} Batch {} Loss {:.4f}'.format(
                                                    int(checkpoint.step),
                                                    batch,
                                                    batch_loss.numpy()))
    total_loss /= (len(X_train) // BATCH_SIZE)
    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    epoch = int(checkpoint.step)
    if int(epoch) % 10 == 0:
        copy_manager.save()
    
    # Calculate validation accuracy
    val_total_loss = 0
    total_accuracy = 0

    for (batch, (inp, targ, tag)) in enumerate(copy_val_dataset):
        batch_loss, batch_accuracy = train_step(inp, targ,
                                                mode='validation',
                                                training=False, 
                                                tag_inp=tag, 
                                                inc_tags=inc_tags)
        val_total_loss += batch_loss
        total_accuracy += batch_accuracy
    
    if epoch % 5 == 0:
        text = lang.sequences_to_texts([input_tensor_train[0]])[0]
        tags = 'COPY'
        text = text.replace(' ', '')
        output(text[1:-1], 'warmup-' + str(epoch), tags=tags, inc_tags=inc_tags)

    val_total_loss /= (len(X_val) // BATCH_SIZE)
    total_accuracy /= (len(X_val))
    
    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    print('Copy: Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            epoch, total_loss, val_total_loss, total_accuracy))
    logger.info('Copy: Epoch {} Loss {:.4f} Validation {:.4f} Val Accuracy {}'.format(
            epoch, total_loss, val_total_loss, total_accuracy))

    if total_accuracy >= 0.75:
        print('Successful. Copying phase over')
        logger.info('Warm-up phase finished with accuracy {}'.format(total_accuracy))
        copy_manager.save()
        break

    if epoch > 20:
        print('Epochs exceeded. Copying phase over')
        logger.info('Warm-up phase breaking with accuracy {}'.format(total_accuracy))
        copy_manager.save()
        break

# Start main phase training
logger.info('Main phase')
loss, val_loss, accuracy = [], [], []
callback = CustomCallback(optimizer, patience=5, cooldown=10)

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (inp, targ, tag)) in enumerate(dataset.take(steps_per_epoch_train)):
        batch_loss, _ = train_step(inp, targ, mode='main',
                                tag_inp=tag, 
                                inc_tags=inc_tags)
        total_loss += batch_loss

        if batch % 100 == 0:
            logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    
    # Update checkpoint step variable and save
    checkpoint.step.assign_add(1)
    if epoch % 10 == 0:
        text = lang.sequences_to_texts([input_tensor_train[0]])[0]
        text = text.replace(' ', '')
        tags = lang.sequences_to_texts([tag_tensor_train[0]])[0]
        output(text[1:-1], 'main-' + str(epoch), tags=tags, inc_tags=inc_tags)
    
    loss.append(total_loss / steps_per_epoch_train)
    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    
    # Calculate validation loss
    val_total_loss = 0
    total_accuracy = 0

    for (batch, (inp, targ, tag)) in enumerate(val_dataset.take(steps_per_epoch_val)):
        batch_loss, batch_accuracy = train_step(inp, targ, mode='validation',
                                                training=False, 
                                                tag_inp=tag, 
                                                inc_tags=inc_tags)
        val_total_loss += batch_loss
        total_accuracy += batch_accuracy
    
    val_loss.append(val_total_loss / steps_per_epoch_val)
    accuracy.append(total_accuracy / len(input_tensor_val))

    print('Epoch {} Loss {:.4f} Validation {:.4f} Validation accuracy {}'.format(
            epoch + 1, loss[-1], val_loss[-1], total_accuracy))
    logger.info('Epoch {} Loss {:.4f} Validation {:.4f} Val Accuracy {}'.format(
            epoch + 1, loss[-1], val_loss[-1], total_accuracy))

    if len(accuracy) > 1 and accuracy[-1] > accuracy[-2]:
        main_manager['accuracy'].save()

    if len(val_loss) > 1 and val_loss[-1] < val_loss[-2]:
        main_manager['validation'].save()
    
    if callback(val_loss[-1]):
        logger.info('Learning rate now {}'.format(callback.get_lr()))
    
    if accuracy[-1] > 0.90:
        print('Main phase breaking')
        logger.info('Main phase finished with accuracy {}'.format(accuracy[-1]))
        main_manager['latest'].save()
        break

# Dump loss values to file
with open(os.path.join(OUT_DIR, 'loss.csv'), 'w') as f:
    f.write('epoch\tloss\tval_loss\taccuracy\n')
    for i, (lo, vo, ac) in enumerate(zip(loss, val_loss, accuracy)):
        f.write('{}\t{}\t{}\t{}\n'.format(i+1, lo, vo, ac))

main_manager['latest'].save()

# Dump weights as well
os.makedirs(os.path.join(OUT_DIR, 'encoder'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'decoder'), exist_ok=True)
encoder.save_weights(os.path.join(OUT_DIR, 'encoder', 'enc-wt'))
decoder.save_weights(os.path.join(OUT_DIR, 'decoder', 'dec-wt'))
if inc_tags:
    os.makedirs(os.path.join(OUT_DIR, 'tag_encoder'), exist_ok=True)
    tag_encoder.save_weights(os.path.join(OUT_DIR, 'tag_encoder', 'tag-enc-wt'))

if main_manager['validation'].latest_checkpoint:
    ckpt = main_manager['validation'].latest_checkpoint
    checkpoint.restore(ckpt)
    print('Restored best validation model')
    logger.debug('Restored best validation model from {}'.format(ckpt))

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

with open(os.path.join(DATA_DIR, 'test.csv'), 'r') as f, \
     open(os.path.join(OUT_DIR, 'test.csv'), 'w') as o:
    corr = 0
    faul = 0
    for i, line in enumerate(f):
        if i >= min(2000, num_samples):
            break
        line = line.strip()
        tag, word, lemma = line.split('\t')
        out, inp = evaluate(word, tag, inc_tags=inc_tags)
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

with open(os.path.join(DATA_DIR, 'train.csv'), 'r') as f, \
     open(os.path.join(OUT_DIR, 'train.csv'), 'w') as o:
    corr = 0
    faul = 0
    for i, line in enumerate(f):
        if i >= min(2000, num_samples):
            break
        line = line.strip()
        tag, word, lemma = line.split('\t')
        out, inp = evaluate(word, tag, inc_tags=inc_tags)
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
