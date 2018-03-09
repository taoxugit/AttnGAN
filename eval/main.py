#!flask/bin/python
import os
import time
from eval import *
from flask import Flask, jsonify, request, abort
from miscc.config import cfg, cfg_from_file

app = Flask(__name__)

# load configuration
cfg_from_file('eval_bird.yml')
# load word dictionaries
wordtoix, ixtoword = word_index()
# lead models
text_encoder, netG = models(len(wordtoix))
# load blob service
blob_service = BlockBlobService(account_name='attgan', account_key=os.environ["BLOB_KEY"])

@app.route('/api/v1.0/bird', methods=['POST'])
def create_bird():
    if not request.json or not 'caption' in request.json:
        abort(400)
    caption = request.json['caption']

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
    return jsonify({'bird': response}), 201

@app.route('/', methods=['GET'])
def get_bird():
    return 'hello!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)