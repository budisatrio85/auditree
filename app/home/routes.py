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

@blueprint.route('/index')
@login_required
def index():
    files = os.listdir(current_app.config['UPLOAD_PATH'])
    text_raw = ""
    text_network_object = ""
    return render_template('index.html', files=files, text_raw=text_raw, text_network_object=text_network_object)
    
@blueprint.route('/index/<filename>')
@login_required
def index_filename(filename):
    files = os.listdir(current_app.config['UPLOAD_PATH'])
    conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
    c = conn.cursor()
    c.execute("SELECT text_raw,text_network_object FROM corpus where filename = ?", [filename])
    data=c.fetchall()
    if len(data)!=0:
        text_raw = data[0][0]
        text_network_object = json.loads(data[0][1])
    conn.commit()
    conn.close()
    return render_template('index.html', files=files, text_raw=text_raw,text_network_object=text_network_object)

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
