from __future__ import print_function

import os
import sys
from nltk.tokenize import RegexpTokenizer

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

if __name__ == "__main__":
    caption = "this bird has a white belly and breast with a short pointy bill"
    
    # load word to index dictionary
    wordtoix = pickle.load(open('data/captions.pickle', 'rb'))[3]
    print(wordtoix)

    # create caption vector
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(caption.lower())
    cap_v = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])
    print(cap_v)


    #gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    print(caption)