#!flask/bin/python
import os
import time
from eval import *
from flask import Flask, jsonify, request, abort
from miscc.config import cfg
from werkzeug.contrib.profiler import ProfilerMiddleware

app = Flask(__name__)


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

    response = eval(request.json['caption'])

    return jsonify({'bird': response}), 201

@app.route('/', methods=['GET'])
def get_bird():
    return 'hello!'

if __name__ == '__main__':
    app.config['PROFILE'] = True
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
    app.run(host='0.0.0.0', port=8080, debug=True)