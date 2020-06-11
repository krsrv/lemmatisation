import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import re
import io
import pickle

def preprocess_sentence(w):
    w = w.strip()
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (ऀ-ॏ, ०-९ ".", "-")
    w = re.sub(r'[^\u0900-\u094f\u0966-\u096f?,.-]+', "", w)

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # START_TOK, END_TOK = '<', '>'
    w = '<' + w + '>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [Word, Lemma]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split()[1:]]  
        for l in lines[:num_examples]]

    return zip(*word_pairs)

def tokenize(lang, tokenizer=None):
    lang_tokenizer = tokenizer
    if lang_tokenizer is None:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False, char_level=True)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
        padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    # since we are working on the same language, the same char set
    # works for both input and target language
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, lang_tokenizer = tokenize(targ_lang, inp_lang_tokenizer)

    return input_tensor, target_tensor, lang_tokenizer

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
