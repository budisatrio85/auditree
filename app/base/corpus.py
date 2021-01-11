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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from string import digits
import tabula
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
pd.set_option('display.max_colwidth', None)
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
        if(length>2):
            cut_list_sorted = list_sorted_distinct_weight[-int(length*0.8):]
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
        
    def convert_pdf_to_string(file_path):
        output_string = StringIO()
        page_and_content = []
        text_content = []
        with open(file_path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            
            if not doc.is_extractable:
                raise PDFTextExtractionNotAllowed

            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            page_no = 0
            content_arr = []
            for pageNumber, page in enumerate(PDFPage.create_pages(doc)):
                if pageNumber == page_no:
                    interpreter.process_page(page)
                    data = output_string.getvalue()
                    page_and_content = [pageNumber+1, data]
                    content_arr.append(data)
                    data = ''
                    output_string.truncate(0)
                    output_string.seek(0)
                    text_content.append(page_and_content)
                page_no = page_no + 1
        
        content_df = pd.DataFrame(text_content, columns = ['Page', 'Content']) 
        content_df['Content'] = content_df['Content'].astype('string')
        # print(df)
        return content_df, content_arr

    def extract_paragraph(page_no, text):
        paragraph = re.split('\.\s{3,}|\s{4,}|\n{3,}|\:\s\n{1,}|\;\s\n{2,}', text)
        temp_arr = []
        df_data = []
        for x in paragraph:
            if isinstance(x, str) and len(x) > 64:
                temp_arr = [page_no, x]
                df_data.append(temp_arr)
            
        df = pd.DataFrame(df_data, columns=['Page', 'Content'])
        df['Content'] = df['Content'].astype('string')
        return df

    def extract_sentences(page_no, text):
        print ("this's for extracting sentences")
        sentence = sent_tokenize(text)
        temp_arr = []
        df_data = []
        for x in sentence:
            if isinstance(x, str) and len(x) > 64:
                temp_arr = [page_no, x]
                df_data.append(temp_arr)
            
        df = pd.DataFrame(df_data, columns=['Page', 'Content'])
        df['Content'] = df['Content'].astype('string')
        return df

    def get_text_from_pdf(file_path, format_result='paragraph'):
        output, output_2 = Corpus.convert_pdf_to_string(file_path)
        df_frames = []
        for x in range(0, len(output.index)):
            text = (output.loc[output['Page'] == x+1, 'Content']).to_string(index=False)
            if format_result == 'paragraph' :
                df = Corpus.extract_paragraph(x+1, text)
            else: 
                df = Corpus.extract_sentences(x+1, text)

            df_frames.append(df)

        data_df = pd.concat(df_frames)
        json_data = data_df.to_json(orient="split")
        return json_data
        
    def process(path_file):
        text = ""
        filename = secure_filename(path_file)
        if filename != '':
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT text_raw,text_network_object FROM corpus WHERE filename = ?", [filename])
            data=c.fetchall()
            if len(data)!=0:
                file_path = os.path.join(current_app.config['UPLOAD_PATH'], filename)
                if list(data[0])[0] is None:
                    # content raw
                    content_text = Corpus.get_text_from_pdf(file_path)
                    c.execute("UPDATE corpus SET text_raw = ? WHERE filename = ?",[content_text,filename])
                
                if list(data[0])[1] is None:
                    # text network
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
                    text = text.translate({ord(k): None for k in digits})
                    
                    # stemming
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    text = stemmer.stem(text)
                    
                    # remover
                    factory = StopWordRemoverFactory()
                    stopword = factory.create_stop_word_remover()
                    text = stopword.remove(text)
                    
                    # process graph
                    graph = Corpus.process_graph(text)
                    c.execute("UPDATE corpus SET text_network_object = ? WHERE filename = ?",[json.dumps(graph),filename])
            conn.commit()
            conn.close()
        return redirect(url_for('home_blueprint.index_filename',filename=filename))