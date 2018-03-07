from __future__ import print_function

import os
import sys
from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, CNN_ENCODER


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

if __name__ == "__main__":
    caption = "this bird has a white belly and breast with a short pointy bill"
    
    # load configuration
    cfg_from_file('eval_bird.yml')

    # load word to index dictionary
    x = pickle.load(open('data/captions.pickle', 'rb'))
    wordtoix = x[3]
    del x

    # create caption vector
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(caption.lower())
    cap_v = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])
    print(cap_v)

    # run inference
    n_words = len(wordtoix)
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)

    #gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    print(caption)