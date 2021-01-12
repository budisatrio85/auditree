# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for, current_app, session,jsonify
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound

import os
import sqlite3
import json
import ast

@blueprint.route('/index')
@login_required
def index():
    files = os.listdir(current_app.config['UPLOAD_PATH'])
    text_raw = ""
    text_network_object = ""
    found_arr = ""
    expected_arr = ""
    filename = ""
    return render_template('index.html', files=files, text_raw=text_raw, text_network_object=text_network_object, found_arr=found_arr, expected_arr=expected_arr, filename=filename)
    
@blueprint.route('/index/<filename>')
@login_required
def index_filename(filename):
    files = os.listdir(current_app.config['UPLOAD_PATH'])
    text_raw = ""
    text_network_object = ""
    found_arr = ""
    expected_arr = ""
    content_raw = []
    conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
    c = conn.cursor()
    c.execute("SELECT text_raw,text_network_object,benford_object FROM corpus where filename = ?", [filename])
    data=c.fetchall()
    if len(data)!=0:
        if list(data[0])[0] is not None:
            text_raw = json.loads(data[0][0])
            for i,item in enumerate(text_raw["index"]):
                content_raw.append({"index":item,"id":text_raw["data"][i][0],"page":text_raw["data"][i][1],"content":text_raw["data"][i][2]})
        if list(data[0])[1] is not None:
            text_network_object = json.loads(data[0][1])
        if list(data[0])[2] is not None:
            benford_object = json.loads(data[0][2])
            found_arr = []
            expected_arr = []
            list_index = benford_object["index"]
            for i,item in enumerate(list_index):
                found_arr.append(benford_object["data"][i][1]*100)
                expected_arr.append(benford_object["data"][i][2]*100)
    conn.commit()
    conn.close()
    return render_template('index.html', files=files, text_raw=content_raw,text_network_object=text_network_object, found_arr=sorted(found_arr,reverse=True), expected_arr=sorted(expected_arr,reverse=True), filename=filename)

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith( '.html' ):
            template += '.html'

        return render_template( template )

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500
