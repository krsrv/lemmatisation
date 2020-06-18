import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import re
import io
import pickle

def preprocess_tags(t):
    t = t.strip()
    t = t.replace(';', ' ')
    t = 'START' + ' ' + t + ' ' + 'END'
    return t

def preprocess_sentence(w, clip_length=None):
    w = w.strip()
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (ऀ-ॏ, ०-९ ".", "-")
    w = re.sub(r'[^\u0900-\u094f\u0958-\u096f?,.-]+', "", w)
    
    if clip_length:
        w = w[-clip_length:]

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # START_TOK, END_TOK = '<', '>'
    w = '<' + w + '>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [Word, Lemma]
def create_dataset(path, num_examples, clip_length):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = []
    for l in lines[:num_examples]:
        ws = l.split('\t')
        word_pairs.append([
            preprocess_tags(ws[0]), 
            preprocess_sentence(ws[1]),
            preprocess_sentence(ws[2])
        ])
    return zip(*word_pairs)

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
        tag_tokenizer.fit_on_texts(tags)

    tensor = tag_tokenizer.texts_to_sequences(tags)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
        padding='post')

    return tensor, tag_tokenizer

# Load dataset
#    If inc_tags, both language and tag tokenizer will be returned. In case tokenizer is
# supplied, both language and tag tokenizer should be given as a single tuple
#    If not inc_tags, only language tokenizer will be returned. In case tokenizer is
# supplied, only language tokenizer is required
def load_dataset(path, num_examples=None, clip_length=None, tokenizer=None, inc_tags=False):
    lang_tokenizer = None

    if tokenizer is None:
        # Load all devanagari characters in tokenizer
        chars = []
        chars.append(''.join([chr(x) for x in range(ord('\u0900'),ord('\u094f'))]))
        chars.append(''.join([chr(x) for x in range(ord('\u0958'),ord('\u096f'))]))
        chars.append('?.,-')  # Punctuation marks
        chars.append('<>')  # Start and end tokens

        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
                filters='', lower=False, char_level=True)
        lang_tokenizer.fit_on_texts(chars)

    if inc_tags:
        tags, inp_lang, targ_lang = create_dataset(path, num_examples, clip_length)
        if tokenizer is not None:
            lang_tokenizer, tag_tokenizer = tokenizer
        else:
            tag_tokenizer = None

        # since we are working on the same language, the same char set
        # works for both input and target language
        input_tensor, _ = tokenize(inp_lang, lang_tokenizer)
        target_tensor, _ = tokenize(targ_lang, lang_tokenizer)

        if tag_tokenizer is None:
            tag_tensor, tag_tokenizer = tag_tokenize(tags)
        else:
            tag_tensor, _ = tag_tokenize(tags, tag_tokenizer)

        return (input_tensor, target_tensor, tag_tensor), (lang_tokenizer, tag_tokenizer)
    else:
        # creating cleaned input, output pairs
        _, inp_lang, targ_lang = create_dataset(path, num_examples, clip_length)    
        if tokenizer is not None:
            lang_tokenizer = tokenizer

        # since we are working on the same language, the same char set
        # works for both input and target language
        input_tensor, _ = tokenize(inp_lang, lang_tokenizer)
        target_tensor, _ = tokenize(targ_lang, lang_tokenizer)

        return (input_tensor, target_tensor), lang_tokenizer

def save_tokeniser(tokeniser, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokeniser, handle)

def load_tokeniser(file_path):
    tokeniser = None
    with open(file_path, 'rb') as handle:
        tokeniser = pickle.load(handle)
    return tokeniser

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def get_number_training_vars(module):
    return np.sum([np.prod(v.get_shape().as_list()) for v in module.trainable_variables])
