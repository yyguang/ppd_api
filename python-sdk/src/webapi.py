#coding=utf-8
'''
Created on 2016年12月22日
@author: yulei
'''

import os
import io
from flask import (Flask, make_response, request, redirect)
from __init__ import BidMount
# from statistics.collect_user_website_from_raw_csv import StatAnalysis

UPLOAD_FOLDER = './uploads'
root_path = './'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/test/')
@app.route('/test/<name>')
@app.route('/test/<string:name>')
@app.route('/test/<string:name>/<int:b>')
@app.route('/test/<string:name>/<int:b>/<string:c>')
def test(name="", b=0, c="c"):
    s1 = "/ split mode result: test=%s,b=%d,c=%s" % (name, b, str(c))
    dt = request.args.to_dict()
    s2 = "? split mode result: %s" % (dt)
    ss = "<br> %s </br> <br> %s </br> " % (s1, s2)
    return ss


@app.route('/about')
def about():
    return 'The about page'


def _save2file(filename, instr, mode='w', create_path=False):
    path = os.path.dirname(filename)
    if create_path and path:
        if not os.path.exists(path):
            os.mkdir(path)

    f = open(filename, mode, encoding='utf-8')
    f.write(instr)
    f.close()


@app.route('/')
@app.route('/hituser_upload')
def upload():
    f = io.open(os.path.join(root_path, 'bidUI.html'), encoding='utf-8')
    data = f.read()
    f.close()
    return data


@app.route('/transform', methods=["POST"])
def transform_handler():
    mount1 = request.values.get("mount1")
    try:
        proc_num = int(mount1)
    except:
        proc_num = 0

    mount2 = request.values.get("mount2")
    try:
        mount2 = int(mount2)
    except:
        mount2 = 0

    mount3 = request.values.get("mount3")
    try:
        mount3 = int(mount3)
    except:
        mount3 = 0

        mount4 = request.values.get("mount4")
    try:
        mount4 = int(mount4)
    except:
        mount4 = 0

    mount5 = request.values.get("mount5")
    try:
        mount5 = int(mount5)
    except:
        mount5 = 0

    email = request.values.get("email")
    try:
        email = str(email)
    except:
        email = None
    input = {'mount1':mount1, 'mount2':mount2, 'mount3':mount3, 'mount4':mount4, 'mount5':mount5, 'email':email}

    bid_parameter = BidMount(input)
    bid_parameter.bidmount()
    return 'Your request has been send!'


if __name__ == "__main__":
    app.debug = True
    app.run('0.0.0.0', 8888, threaded=True)
