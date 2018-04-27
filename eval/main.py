import os
import time
import random
from eval import *
from flask import Flask, jsonify, request, abort
from applicationinsights import TelemetryClient
from applicationinsights.requests import WSGIApplication
from applicationinsights.exceptions import enable
from miscc.config import cfg
#from werkzeug.contrib.profiler import ProfilerMiddleware

enable(os.environ["TELEMETRY"])
app = Flask(__name__)
app.wsgi_app = WSGIApplication(os.environ["TELEMETRY"], app.wsgi_app)

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

@app.route('/api/v1.0/birds', methods=['POST'])
def create_birds():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG, blob_service, copies=6)
    t1 = time.time()

    response = {
        'bird1' : { 'small': urls[0], 'medium': urls[1], 'large': urls[2] },
        'bird2' : { 'small': urls[3], 'medium': urls[4], 'large': urls[5] },
        'bird3' : { 'small': urls[6], 'medium': urls[7], 'large': urls[8] },
        'bird4' : { 'small': urls[9], 'medium': urls[10], 'large': urls[11] },
        'bird5' : { 'small': urls[12], 'medium': urls[13], 'large': urls[14] },
        'bird6' : { 'small': urls[15], 'medium': urls[16], 'large': urls[17] },
        'caption': caption,
        'elapsed': t1 - t0
    }
    return jsonify({'bird': response}), 201

@app.route('/', methods=['GET'])
def get_bird():
    return 'Version 1'

if __name__ == '__main__':
    t0 = time.time()
    tc = TelemetryClient(os.environ["TELEMETRY"])
    
    # gpu based
    cfg.CUDA = os.environ["GPU"].lower() == 'true'
    tc.track_event('container initializing', {"CUDA": str(cfg.CUDA)})

    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service
    blob_service = BlockBlobService(account_name='attgan', account_key=os.environ["BLOB_KEY"])

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(seed)

    #app.config['PROFILE'] = True
    #app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
    #app.run(host='0.0.0.0', port=8080, debug = True)

    t1 = time.time()
    tc.track_event('container start', {"starttime": str(t1-t0)})
    app.run(host='0.0.0.0', port=8080)
