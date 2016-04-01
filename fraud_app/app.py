from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import cPickle as pickle
import pandas as pd
import sys
from datetime import datetime
import socket
import json
import requests
import socket
import time
from parse_live_data import FraudModel
from pandas.io.json import json_normalize
from collections import Counter

PORT = 8000
REGISTER_URL = "http://10.6.2.185:5000/register"
DATA = []
TIMESTAMP = []
STORED_DATA = []
STORED_COUNT = []

def parse_newdata(data):
    predictions = []
    for d in data:
        jdoc = json.loads(d)
        newdf = json_normalize(jdoc)
        fm = FraudModel(newdf)
        processed_df = fm.preprocess_data()
        processed_df.fillna(0, inplace=True)
        predictions.append(run_model(processed_df))
    return predictions, processed_df, jdoc, newdf


def run_model(df):
    with open('data/model6_final.pkl') as f:
        model = pickle.load(f)
        return model.predict(df)[0]

# will register the server address and retrieve json document
def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)

# grabs json document and stores in DATA
@app.route('/score', methods=['POST'])
def score():
    DATA.append(json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())
    return ""

@app.route('/check')
def check():
    prediction = ''
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
        prediction, newdf, jdoc, newdf = parse_newdata(DATA)
        data = [line2, line1, prediction, jdoc, Counter(prediction)]
        if data not in STORED_DATA:
            STORED_DATA.append(data)
    else:
        output = line1

    if prediction != '':
        return render_template('predict.html', prediction=prediction, df=newdf.T.to_html(), date=line2, length=line1, jdoc=jdoc, stored_data =STORED_DATA)
    else:
        return render_template('nopredict.html', output=output)



# home page
@app.route('/')
@app.route('/home')
def index():
    names = ["Muneeb,","Jennifer,","Jesse,","Tim"]
    return render_template('index.html', names=names)

# predict page
@app.route('/predict', methods=['POST'])
def prediction_page():
    text = request.form['user_input'].encode('ascii','ignore')
    prediction = run_model(text)
    return render_template('predict.html', prediction=prediction, text=text)

# model page
@app.route('/model')
def model():
    names = ["Muneeb,","Jennifer,","Jesse,","Tim"]
    return render_template('feat_impt.html', names=names)


if __name__ == '__main__':
    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print "attempting to register %s:%d" % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
