import os
import PyPDF2
import sqlite3
import networkx as nx
import json
import random
from networkx.readwrite import json_graph
from flask import jsonify
from flask import redirect, current_app, url_for, session
from werkzeug.utils import secure_filename

class Corpus():
    def rescale(values, new_min = 0, new_max = 10):
        output = []
        old_min, old_max = min(values), max(values)
        for v in values:
            new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
            output.append(new_v)
        return output
    def generate_random_color(N=5):
        color_list = []
        for x in range(1,N):
            color_list.append("#%06x" % random.randint(0, 0xFFFFFF))
        return color_list
    def process_graph(text):
        G = nx.Graph()
        word_list = []
        text = (text.strip()).split()
        for i, word in enumerate(text):
            if i != len(text)-1:
                word_a = word
                word_b = text[i+1]
                if word_a not in word_list:
                    word_list.append(word_a)
                if word_b not in word_list:
                    word_list.append(word_b)
                if G.has_edge(word_a,word_b):
                    G[word_a][word_b]['weight'] += 1
                else:
                    G.add_edge(word_a,word_b, weight = 1)
        
        list_weight = [int(row[1]["weight"]) for row in dict(G.edges).items()] 
        cut_list_sorted = list_sorted_distinct_weight = sorted(list(set(list_weight)))
        length = len(list_sorted_distinct_weight)
        #if(length>2):
        #    cut_list_sorted = list_sorted_distinct_weight[-int(length/2):]
        min_weight = min(cut_list_sorted)
        max_weight = max(cut_list_sorted)
        jsonres = json_graph.node_link_data(G)
        data = jsonres["links"]
        node = jsonres["nodes"]
        data_list = []
        node_list = []
        color_list = Corpus.generate_random_color(12)
        for i,item in enumerate(data):
            if item["weight"] > min_weight:
                data_list.append([item["source"],item["target"]]) 
                found = False
                val_weight = 10 / (max_weight - min_weight) * (item["weight"] - min_weight) + 1
                for j,key in enumerate(node_list): 
                    if key["id"] == item["source"]:
                        found = True
                        if int(val_weight*5) > key["marker"]["radius"]:
                            node_list[j]["marker"]["radius"] = int(val_weight*5)  
                            node_list[j]["color"] = color_list[int(val_weight)-1]
                if not found:
                    node_list.append({"id":item["source"],"marker": {"radius": int(val_weight*5),},"color":color_list[int(val_weight)-1]})
        return {"data":data_list,"node":node_list, "min_weight":min_weight}
    def process(path_file):
        text = ""
        filename = secure_filename(path_file)
        if filename != '':
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT text_raw FROM corpus WHERE filename = ?", [filename])
            data=c.fetchall()
            if len(data)!=0 and data[0] == (None,):
                file_path = os.path.join(current_app.config['UPLOAD_PATH'], filename)
                reader = PyPDF2.PdfFileReader(file_path)
                for page in reader.pages:  
                    text += page.extractText()
                    
                # cleaning
                text = text.replace("\n",' ')
                text = text.replace(":",' ')
                text = text.replace(".",' ')
                text = text.replace("/",' ')
                text = text.replace(",",' ')
                text = text.replace("(",' ')
                text = text.replace(")",' ')
                text = text.lower()
                text = ' '.join(text.split())
                c.execute("UPDATE corpus SET text_raw = ? WHERE filename = ?",[text,filename])
                
                # process graph
                graph = Corpus.process_graph(text)
                c.execute("UPDATE corpus SET text_network_object = ? WHERE filename = ?",[json.dumps(graph),filename])
            conn.commit()
            conn.close()
        return redirect(url_for('home_blueprint.index_filename',filename=filename))