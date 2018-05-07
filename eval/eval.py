from __future__ import print_function

import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
from azure.storage.blob import BlockBlobService

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    #print(captions.astype(int), cap_lens.astype(int))
    #captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])
    #print(captions, cap_lens)
    #return captions, cap_lens

    return captions.astype(int), cap_lens.astype(int)

def generate(caption, wordtoix, ixtoword, text_encoder, netG, blob_service, copies=2):
    # load word vector
    captions, cap_lens  = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)

    # only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), volatile=True)
    cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

    if cfg.CUDA:
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        noise = noise.cuda()

    

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

    # ONNX EXPORT
    #export = os.environ["EXPORT_MODEL"].lower() == 'true'
    if False:
        print("saving text_encoder.onnx")
        text_encoder_out = torch.onnx._export(text_encoder, (captions, cap_lens, hidden), "text_encoder.onnx", export_params=True)
        print("uploading text_encoder.onnx")
        blob_service.create_blob_from_path('models', "text_encoder.onnx", os.path.abspath("text_encoder.onnx"))
        print("done")

        print("saving netg.onnx")
        netg_out = torch.onnx._export(netG, (noise, sent_emb, words_embs, mask), "netg.onnx", export_params=True)
        print("uploading netg.onnx")
        blob_service.create_blob_from_path('models', "netg.onnx", os.path.abspath("netg.onnx"))
        print("done")
        return

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    # storing to blob storage
    container_name = "images"
    full_path = "https://attgan.blob.core.windows.net/images/%s"
    prefix = datetime.now().strftime('%Y/%B/%d/%H_%M_%S_%f')
    urls = []
    # only look at first one
    #j = 0
    for j in range(batch_size):
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
            if copies > 2:
                blob_name = '%s/%d/%s_g%d.png' % (prefix, j, "bird", k)
            else:
                blob_name = '%s/%s_g%d.png' % (prefix, "bird", k)
            blob_service.create_blob_from_stream(container_name, blob_name, stream)
            urls.append(full_path % blob_name)

            if copies == 2:
                for k in range(len(attention_maps)):
                #if False:
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

                        blob_name = '%s/%s_a%d.png' % (prefix, "attmaps", k)
                        blob_service.create_blob_from_stream(container_name, blob_name, stream)
                        urls.append(full_path % blob_name)
        if copies == 2:
            break
    
    #print(len(urls), urls)
    return urls

def word_index():
    ixtoword = cache.get('ixtoword')
    wordtoix = cache.get('wordtoix')
    if ixtoword is None or wordtoix is None:
        #print("ix and word not cached")
        # load word to index dictionary
        x = pickle.load(open('data/captions.pickle', 'rb'))
        ixtoword = x[2]
        wordtoix = x[3]
        del x
        cache.set('ixtoword', ixtoword, timeout=60 * 60 * 24)
        cache.set('wordtoix', wordtoix, timeout=60 * 60 * 24)

    return wordtoix, ixtoword

def models(word_len):
    #print(word_len)
    text_encoder = cache.get('text_encoder')
    if text_encoder is None:
        #print("text_encoder not cached")
        text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        if cfg.CUDA:
            text_encoder.cuda()
        text_encoder.eval()
        cache.set('text_encoder', text_encoder, timeout=60 * 60 * 24)

    netG = cache.get('netG')
    if netG is None:
        #print("netG not cached")
        netG = G_NET()
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        if cfg.CUDA:
            netG.cuda()
        netG.eval()
        cache.set('netG', netG, timeout=60 * 60 * 24)

    return text_encoder, netG

def eval(caption):
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service
    blob_service = BlockBlobService(account_name='attgan', account_key=os.environ["BLOB_KEY"])

    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG, blob_service)
    t1 = time.time()

    response = {
        'small': urls[0],
        'medium': urls[1],
        'large': urls[2],
        'map1': urls[3],
        'map2': urls[4],
        'caption': caption,
        'elapsed': t1 - t0
    }

    return response

if __name__ == "__main__":
    caption = "the bird has a yellow crown and a black eyering that is round"
    
    # load configuration
    #cfg_from_file('eval_bird.yml')
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service
    blob_service = BlockBlobService(account_name='attgan', account_key='[REDACTED]')
    
    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG, blob_service)
    t1 = time.time()
    print(t1-t0)
    print(urls)