import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as mfm

import re
import io
import pickle

def preprocess_tags(t):
    t = t.strip()
    t = t.replace(';', ' ')
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
# 3. Return word pairs in the format: [Tag, Word, Lemma]
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

# Load dataset
#    If inc_tags, both language and tag tokenizer will be returned. In case tokenizer is
# supplied, both language and tag tokenizer should be given as a single tuple
#    If not inc_tags, only language tokenizer will be returned. In case tokenizer is
# supplied, only language tokenizer is required
def load_dataset(path, num_examples=None, clip_length=None, tokenizer=None):
    lang_tokenizer, tag_tokenizer = None, None

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

    tags, inp_lang, targ_lang = create_dataset(path, num_examples, clip_length)
    if tokenizer is not None:
        lang_tokenizer, tag_tokenizer = tokenizer
    
    # since we are working on the same language, the same char set
    # works for both input and target language
    input_tensor, _ = tokenize(inp_lang, lang_tokenizer)
    target_tensor, _ = tokenize(targ_lang, lang_tokenizer)

    if tag_tokenizer is None:
        tag_tensor, tag_tokenizer = tag_tokenize(tags)
    else:
        tag_tensor, _ = tag_tokenize(tags, tag_tokenizer)

    return (input_tensor, target_tensor, tag_tensor), (lang_tokenizer, tag_tokenizer)

def save_tokeniser(tokeniser, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokeniser, handle)

def load_tokeniser(file_path):
    tokeniser = None
    with open(file_path, 'rb') as handle:
        tokeniser = pickle.load(handle)
    return tokeniser

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

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence, fname=None):
    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    prop = mfm.FontProperties(fname='Nirmala.ttf')
    # fontdict = {'fontsize': 16}
    ax.tick_params(labelsize=16)
    
    ax.xaxis.set_label_text('Input')
    ax.yaxis.set_label_text('Output')
    
    ax.set_xticklabels('S'+sentence, fontproperties=prop, rotation=90)
    ax.set_yticklabels('S'+predicted_sentence, fontproperties=prop)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)

def get_number_training_vars(module):
    return np.sum([np.prod(v.get_shape().as_list()) for v in module.trainable_variables])
