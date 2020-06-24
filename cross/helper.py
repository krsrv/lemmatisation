import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from  matplotlib.font_manager import FontProperties

import re
import io
import os
import pickle

def preprocess_tags(t):
    t = t.strip()
    t = t.replace(';', ' ')
    return t

def preprocess_sentence(w, clip_length=None):
    w = w.strip()
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (ऀ-ॏ, ०-९ ".", "-")
    # w = re.sub(r'[^\u0900-\u094f\u0958-\u096f?,.-]+', "", w)
    
    if clip_length:
        w = w[-clip_length:]

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # START_TOK, END_TOK = '<', '>'
    w = '<' + w + '>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [Tag, Word, Lemma]
# Note: SIGMORPHON 2019 Task 2 has data in the form Lemma, Word, Tag
# Here we use files with the format Tag, Word, Lemma
def create_dataset(path, num_examples=None, clip_length=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = []
    for l in lines[:num_examples]:
        ws = l.split('\t')
        word_pairs.append([
            preprocess_tags(ws[0]), 
            preprocess_sentence(ws[1], clip_length=clip_length),
            preprocess_sentence(ws[2], clip_length=clip_length)
        ])
    return zip(*word_pairs)

def create_tokenizer(*files):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False, char_level=True)
    tag_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False)
    
    word_pairs = []
    
    for File in files:
        lines = io.open(File, encoding='UTF-8').read().strip().split('\n')
        word_pairs = []
        for l in lines:
            ws = l.split('\t')
            word_pairs.append([
                preprocess_tags(ws[0]), 
                preprocess_sentence(ws[1]),
                preprocess_sentence(ws[2])
            ])

    tags, words, lemmas = zip(*word_pairs)
    tag_tokenizer.fit_on_texts(tags + ('COPY',))
    lang_tokenizer.fit_on_texts(words + lemmas)

    return (lang_tokenizer, tag_tokenizer)

def tokenize(lang, lang_tokenizer=None):
    # If tokenizer is supplied, assume that it is already fit to the input lang
    if lang_tokenizer is None:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False, char_level=True)
        lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
        padding='post')

    return tensor, lang_tokenizer

def tag_tokenize(tags, tag_tokenizer=None):
    # If tokenizer is supplied, assume that it is already fit to the input lang
    if tag_tokenizer is None:
        tag_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False)
        # Add a new 'COPY' tag as well
        tag_tokenizer.fit_on_texts(tags + ('COPY',))

    tensor = tag_tokenizer.texts_to_sequences(tags)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
        padding='post')

    return tensor, tag_tokenizer

def load_ptv(path, dim=None, num_examples=None):
    array = np.loadtxt(path, dtype=np.float32)
    return array[:num_examples]

# Load dataset
#    If inc_tags, both language and tag tokenizer will be returned. In case tokenizer is
# supplied, both language and tag tokenizer should be given as a single tuple
#    If not inc_tags, only language tokenizer will be returned. In case tokenizer is
# supplied, only language tokenizer is required
def load_dataset(path, num_examples=None, clip_length=None, tokenizer=None):
    lang_tokenizer, tag_tokenizer = None, None

    tags, inp_lang, targ_lang = create_dataset(path, num_examples, clip_length)
    if tokenizer is not None:
        if len(tokenizer) == 2:
            lang_tokenizer, tag_tokenizer = tokenizer
        else:
            lang_tokenizer = tokenizer
    
    # since we are working on the same language, the same char set
    # works for both input and target language
    input_tensor, _ = tokenize(inp_lang, lang_tokenizer)
    target_tensor, _ = tokenize(targ_lang, lang_tokenizer)

    if tag_tokenizer is None:
        tag_tensor, tag_tokenizer = tag_tokenize(tags)
    else:
        tag_tensor, _ = tag_tokenize(tags, tag_tokenizer)

    return (input_tensor, target_tensor, tag_tensor), (lang_tokenizer, tag_tokenizer)

# Path to directory with 'train.csv', 'dev.csv', 'test.csv'
def load_ttd_files(path, num_examples=None, clip_length=None, tokenizer=None):
    train_file = os.path.join(path, 'train.csv')
    test_file = os.path.join(path, 'test.csv')
    val_file = os.path.join(path, 'val.csv')

    if tokenizer is None:
        tokenizer = create_tokenizer(train_file, test_file, val_file)
    train_tensors, _ = load_dataset(train_file, num_examples=num_examples, 
                                                   clip_length=clip_length, tokenizer=tokenizer)
    test_tensors, _ = load_dataset(test_file, num_examples=num_examples, 
                                                 clip_length=clip_length, tokenizer=tokenizer)
    val_tensors, _ = load_dataset(val_file, num_examples=num_examples, 
                                               clip_length=clip_length, tokenizer=tokenizer)

    return train_tensors, test_tensors, val_tensors, tokenizer

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle)

def load_tokenizer(file_path):
    tokenizer = None
    with open(file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def pad(x, y, concatenate=True):
    # Pad x and y (2D tensors) so that they are the same sequence length
    # and concatenate them along axis=0
    maxlen = max(x.shape[1], y.shape[1])
    x = tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                      maxlen=maxlen, 
                                                      padding='post')
    y = tf.keras.preprocessing.sequence.pad_sequences(y, 
                                                      maxlen=maxlen, 
                                                      padding='post')
    if concatenate:
        return np.concatenate([x, y], axis=0)
    else:
        return x, y

def create_copy_dataset_from_tensors(input_tensor, target_tensor, tag_tensor, copy_tag=None):
    assert copy_tag is not None
    X = pad(input_tensor, target_tensor, concatenate=True)
    Y = X[:, :]
    
    copy_tag_tensor = np.reshape(copy_tag, (1,1))
    copy_tag_tensor = tf.keras.preprocessing.sequence.pad_sequences(copy_tag_tensor, 
                                                    maxlen=tag_tensor.shape[1], 
                                                    padding='post')
    copy_tag_tensor = np.repeat(copy_tag_tensor, input_tensor.shape[0], axis=0)
    T = pad(copy_tag_tensor, tag_tensor)

    return X, Y, T

def create_checkpoint_manager(ckpt, path):
    manager = {
        'accuracy': tf.train.CheckpointManager(checkpoint,
            os.path.join(path, 'acc'), max_to_keep=1),
        'validation': tf.train.CheckpointManager(checkpoint,
            os.path.join(path, 'val'), max_to_keep=1),
        'latest': tf.train.CheckpointManager(checkpoint,
            os.path.join(path, 'latest'), max_to_keep=1)
        }
    return manager

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence, fname=None):
    fig = plt.figure(figsize=(10,10))
        
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    prop = FontProperties(fname='Nirmala.ttf')
    
    fontdict = {'fontsize': 14, 'color': 'darkred'}
    sentence = [x + ' ' for x in sentence]
    predicted_sentence = [x + ' ' for x in predicted_sentence]
    
    ax.xaxis.set_label_text('Input')
    ax.yaxis.set_label_text('Output')
    ax.set_xticklabels([''] + sentence, fontproperties=prop, fontdict=fontdict)
    ax.set_yticklabels([''] + predicted_sentence, fontproperties=prop, fontdict=fontdict)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)  

def get_number_training_vars(module):
    return np.sum([np.prod(v.get_shape().as_list()) for v in module.trainable_variables])
