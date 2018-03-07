from __future__ import print_function

import os
import sys
import torch
import numpy as np
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, G_NET

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

if __name__ == "__main__":
    caption = "this bird has a white belly and breast with a short pointy bill"
    
    # load configuration
    cfg_from_file('eval_bird.yml')
    print(cfg.CUDA) # = False

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

    # expected state for single generation
    captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])

    # run inference
    n_words = len(wordtoix)
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()
    netG = G_NET()

    # only one to generate
    batch_size = captions.shape[0]
    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), volatile=True)
    cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = (captions == 0)

    #######################################################
    # (2) Generate fake images
    #######################################################
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()




    print(caption)