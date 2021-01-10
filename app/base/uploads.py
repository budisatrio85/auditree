import os
from flask import redirect, current_app, url_for
from werkzeug.utils import secure_filename
import sqlite3

class Upload():
    def upload_files(uploaded_file):
        #uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            uploaded_file.save(os.path.join(current_app.config['UPLOAD_PATH'], filename))
            
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT filename FROM corpus where filename = ?", [filename])
            data=c.fetchall()
            if len(data)==0:
                c.execute("INSERT INTO corpus (filename) VALUES (?)",[filename])
            conn.commit()
            conn.close()
        return redirect(url_for('home_blueprint.index'))
        
    def delete_files(uploaded_file):
        #uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file)
        if os.path.exists(os.path.join(current_app.config['UPLOAD_PATH'], filename)):
            os.remove(os.path.join(current_app.config['UPLOAD_PATH'], filename))
            
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT filename FROM corpus where filename = ?", [filename])
            data=c.fetchall()
            if len(data)!=0:
                c.execute("DELETE FROM corpus WHERE filename = (?)",[filename])
            conn.commit()
            conn.close()
        return redirect(url_for('home_blueprint.index'))
        
    def process_files(uploaded_file):
        #uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file)
        if os.path.exists(os.path.join(current_app.config['UPLOAD_PATH'], filename)):
            os.remove(os.path.join(current_app.config['UPLOAD_PATH'], filename))
        return redirect(url_for('home_blueprint.index'))