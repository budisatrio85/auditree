# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_migrate import Migrate
from os import environ
from sys import exit
from decouple import config

from config import config_dict
from app import create_app, db

import os
import sqlite3

# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app( app_config ) 
Migrate(app, db)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_PATH'] = "/mnt/d/SOURCE/AudiTREE/document"
app.config['DB_PATH'] = "/mnt/d/SOURCE/AudiTREE/"
app.config['PICKLE_PATH'] = "/mnt/d/SOURCE/AudiTREE/pickle"

conn = sqlite3.connect(os.path.join(app.config['DB_PATH'], 'auditree.db'))

if __name__ == "__main__":
    app.run()
