#!flask/bin/python
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64
import json
#import time
from datetime import datetime

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/upload', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_base64_img():
    content = request.get_json()
    # checking if the image is present or not.
    if 'image' not in content:
        abort(400)
        abort(Response('No Image data received'))

    imgdata = base64.b64decode(content['image'])

    (dt, micro) = datetime.utcnow().strftime('%Y%m%d%H%M%S.%f').split('.')
    dt = "%s%03d" % (dt, int(micro) / 1000)
    filename = 'images/'+dt+'.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    s = int(dt)%65 + 65

    result = {
        "hand_object": str(base64.b64encode(imgdata)),
        "letter_detected": chr(s)
    }

    return json.dumps(result)

app.run(port=5000)