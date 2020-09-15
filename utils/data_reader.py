import os
import random
import numpy as np
import pickle
import json
import argparse
import numpy as np
from collections import Counter
import spacy

from utils import constant
from utils.config import config

nlp = spacy.load("en_core_web_sm")

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.randn(vocab_size, config.embed_dim) * 0.01
    emb[constant.PAD_ID] = 0 # <pad> should be all 0

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb


def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab



class Vocab(object):
    def __init__(self, init_wordlist, word_counter):
        self.word2id = {w:i for i,w in enumerate(init_wordlist)}
        self.id2word = {i:w for i,w in enumerate(init_wordlist)}
        self.n_words = len(init_wordlist)  
        self.word_counter = word_counter

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]
    



def get_feats(utters, word_pairs):
    ret = {'tokens': [], 'dep_head': [], 'dep_tag':[], 'pos_tag':[], 'ner_iob':[], 'ner_type':[], 'noun_chunks':[], 'noun_chunks_root':[]}
    for utter in utters:
        if config.lower: utter = utter.lower()
        break_index = utter.find(':')
        speaker, utter = utter[:break_index], utter[break_index:]
        speaker = ''.join(speaker.split()) # remove white space
        utter = speaker + utter
        # DONE: 1. unsplit speaker 2. ner type -> PER 3. change x and y
        # DONE: pass x and y through nlp
        for k,v in word_pairs.items():
            utter = utter.replace(k,v)
        utter = nlp(utter)
        ret['tokens'].append([str(token) for token in utter])
        ret['dep_head'].append( [token.head.i+1 if token.i != token.head.i else 0 for token in utter ])
        ret['dep_tag'].append([token.dep_ for token in utter])
        ret['pos_tag'].append( [token.pos_ for token in utter])
        ret['ner_iob'].append([utter[i].ent_iob_ for i in range(len(utter))])
        ret['ner_type'].append([utter[i].ent_type_ if i!=0 else 'PERSON' for i in range(len(utter))]) # hard-code ner type to be 'PER' for speaker
        ret['noun_chunks'].append([str(o) for o in utter.noun_chunks])
        ret['noun_chunks_root'].append([str(o.root) for o in utter.noun_chunks])

    return ret
    

word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}


def load_data(filename):
    tokens = []
    word_pairs = {}
    with open(filename) as infile:
        data = json.load(infile)
    D = []
    for i in range(len(data)):
        utters = data[i][0]
        spacy_feats = get_feats(utters, word_pairs)
        for j in range(len(data[i][1])):
            d = {}
            d['us'] = utters
            d['feats'] = spacy_feats                 
            d['x_type'] = data[i][1][j]["x_type"]
            d['y_type'] = data[i][1][j]["y_type"]
            d['rid'] = data[i][1][j]["rid"]
            d['r'] = data[i][1][j]["r"]
            d['t'] = data[i][1][j]["t"]

            d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
            d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
            d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
            d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
            D.append(d)
        
        tokens += [oo for o in d['feats']['tokens'] for oo in o]

    return tokens, D

def load_data_c(filename):
    tokens = []
    word_pairs = {}
    with open(filename) as infile:
        data = json.load(infile)
    D = []
    for i in range(len(data)):
        utters = data[i][0]
        spacy_feats = get_feats(utters, word_pairs)
        for j in range(len(data[i][1])):
            for l in range(1, len(data[i][0])+1):
                d = {}
                d['us'] = utters[:l]
                d['feats'] = {k:v[:l] for k,v in spacy_feats.items()}
                d['x_type'] = data[i][1][j]["x_type"]
                d['y_type'] = data[i][1][j]["y_type"]
                d['rid'] = data[i][1][j]["rid"]
                d['r'] = data[i][1][j]["r"]
                d['t'] = data[i][1][j]["t"]

                d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
                d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
                d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
                d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
                D.append(d)
        
        tokens += [oo for o in d['feats']['tokens'] for oo in o]

    return tokens, D


def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if config.min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v, counter

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched


def load_dataset():
    if os.path.exists(config.proce_f):
        print("LOADING dialogre dataset")
        with open(config.proce_f, "rb") as f: [train_data, dev_data, test_data, vocab] = pickle.load(f)
        return train_data, dev_data, test_data, vocab

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")
    train_tokens, train_data = load_data(config.train_f)
    dev_tokens, dev_data = load_data(config.val_f)
    test_tokens, test_data = load_data(config.test_f)

    # load glove
    print("loading glove...")
    glove_vocab = load_glove_vocab(config.glove_f, config.embed_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")

    # TODO: The vocab should contain all 3 splits? 
    v, v_counter = build_vocab(train_tokens + dev_tokens + test_tokens, glove_vocab, config.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = build_embedding(config.glove_f, v, config.embed_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(config.proce_f, 'wb') as outfile:
        vocab = Vocab(v, v_counter)
        pickle.dump([train_data, dev_data, test_data, vocab], outfile)
    np.save(config.embed_f, embedding)
    print("all done.")

    return train_data, dev_data, test_data, vocab

def load_dataset_c():
    if os.path.exists(config.proce_f_c):
        print("LOADING dialogre dataset for conversation setting")
        with open(config.proce_f_c, "rb") as f: [dev_data, test_data] = pickle.load(f)
        return dev_data, test_data

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")
    dev_tokens, dev_data = load_data_c(config.val_f)
    test_tokens, test_data = load_data_c(config.test_f)

    print("dumping to files...")
    with open(config.proce_f_c, 'wb') as outfile:
        pickle.dump([dev_data, test_data], outfile)
    print("all done.")

    return dev_data, test_data

def get_original_data(fn):
    with open(fn, "r", encoding='utf8') as f:
        data = json.load(f)    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for k in range(len(data[i][1][j]["rid"])):
                data[i][1][j]["rid"][k] -= 1
    return data

if __name__ == "__main__":
    load_dataset()
    load_data_c()
