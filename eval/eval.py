from __future__ import print_function

import os
import sys
import torch
import io
import numpy as np
from PIL import Image
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
from azure.storage.blob import BlockBlobService

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

if __name__ == "__main__":
    caption = "the bird has a yellow crown and a black eyering that is round"
    
    # load configuration
    cfg_from_file('eval_bird.yml')

    # load word to index dictionary
    x = pickle.load(open('data/captions.pickle', 'rb'))
    ixtoword = x[2]
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
    n_words = len(wordtoix)

    # run inference
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.eval()

    print('Load G from: ', cfg.TRAIN.NET_G)
    netG = G_NET()
    model_dir = cfg.TRAIN.NET_G
    state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.eval()

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

    # storing to blob storage
    blob_service = BlockBlobService(account_name='attgan', account_key='tqMJN9RH+MW7UhDQbIEyXxoS1/wqAtvfKiC7hHJ8QrbtWF2k6yeIb/xHtya3QRGmUyDD7pATI4op2Ni6Iji4qQ==')
    container_name = "images"
    full_path = "https://attgan.blob.core.windows.net/images/%s"
    urls = []
    # only look at first one
    j = 0
    for k in range(len(fake_imgs)):
        im = fake_imgs[k][j].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)

        # save image to stream
        stream = io.BytesIO()
        im.save(stream, format="png")
        stream.seek(0)

        blob_name = '%s_g%d.png' % ("bird", k)
        blob_service.create_blob_from_stream(container_name, blob_name, stream)
        urls.append(full_path % blob_name)

    for k in range(len(attention_maps)):
        if len(fake_imgs) > 1:
            im = fake_imgs[k + 1].detach().cpu()
        else:
            im = fake_imgs[0].detach().cpu()
        attn_maps = attention_maps[k]
        att_sze = attn_maps.size(2)
        img_set, sentences = \
            build_super_images2(im[j].unsqueeze(0),
                                captions[j].unsqueeze(0),
                                [cap_lens_np[j]], ixtoword,
                                [attn_maps[j]], att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            stream = io.BytesIO()
            im.save(stream, format="png")
            stream.seek(0)

            blob_name = '%s_a%d.png' % ("attmaps", k)
            blob_service.create_blob_from_stream(container_name, blob_name, stream)
            urls.append(full_path % blob_name)

    print(caption)
    print(urls)