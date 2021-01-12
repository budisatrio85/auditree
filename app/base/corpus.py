import os
import PyPDF2
import sqlite3
import networkx as nx
import json
import random
from networkx.readwrite import json_graph
from flask import jsonify
from flask import redirect, current_app, url_for, session, Response
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
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from scipy import spatial
import faiss
import pickle
from pathlib import Path
import benford as bf

CHARACTER_THRESHOLD = 350
N_RECOMMENDED = 3
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
            if isinstance(x, str) and len(x) > CHARACTER_THRESHOLD:
                if re.search("\.{5,}", x):
                    continue
                else:
                    temp_arr = [random.randint(100, 10000), page_no, x]
                    df_data.append(temp_arr)
            
        df =pd.DataFrame(df_data, columns=['id', 'Page', 'Content'])
        df['Content'] = df['Content'].astype('string')

        return df

    def extract_sentences(page_no, text):
        sentence = sent_tokenize(text)
        temp_arr = []
        df_data = []
        for x in sentence:
            if isinstance(x, str) and len(x) > CHARACTER_THRESHOLD:
                if re.search("\.{5,}", x):
                    continue
                else:
                    temp_arr = [random.randint(100, 10000), page_no, x]
                    df_data.append(temp_arr)
            
        df =pd.DataFrame(df_data, columns=['id', 'Page', 'Content'])
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

    def id2details(df, I, column):
        return [list(df[df.id == idx][column]) for idx in I[0]]

    def id2details_df(df, I):
        mask = df['id'].isin(I[0])
        return df.loc[mask]

    def vector_search_indo(query, model, tokenizer, index, num_results=5):
        query=tokenizer.encode(query,truncation=True,max_length=512, add_special_tokens=True)
        tokens = [query]
        max_len = 0
        for i in tokens:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:,0,:].numpy()
        D, I = index.search(np.array(features).astype("float32"), k=num_results)
        return D, I

    def load_indobert_model():
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
        return model, tokenizer

    def load_faiss_index(path_to_faiss="faiss_index.pickle"):
        """Load and deserialize the Faiss index."""
        with open(path_to_faiss, "rb") as h:
            data = pickle.load(h)
        return faiss.deserialize_index(data)

    def search_paragraph(query,path_file):
        result = ""
        filename = secure_filename(path_file)
        if filename != '':
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT text_raw FROM corpus WHERE filename = ?", [filename])
            data=c.fetchall()
            if len(data)!=0:
                file_path = os.path.join(current_app.config['UPLOAD_PATH'], filename)
                content_raw = list(data[0])[0]
                if content_raw is not None:
                    data = pd.read_json(content_raw,orient="split")
                    model, tokenizer = Corpus.load_indobert_model()
                    PATH_TO_FAISS_PICKLE = os.path.join(current_app.config['PICKLE_PATH'], filename+".pickle")
                    faiss_index = Corpus.load_faiss_index(PATH_TO_FAISS_PICKLE)

                    D, I = Corpus.vector_search_indo(query, model, tokenizer, faiss_index, N_RECOMMENDED)

                    #print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
                    # res = id2details(data, I, 'Content')
                    res = Corpus.id2details_df(data, I)
                    result_json = res.to_json(orient="split")
                    result = json.loads(result_json)
                    result_raw = []
                    for i,item in enumerate(result["index"]):
                        result_raw.append({"index":item,"id":result["data"][i][0],"page":result["data"][i][1],"content":result["data"][i][2]})
                    result = str(result_raw)
            conn.commit()
            conn.close()
        return result
        #return Response(result, mimetype='application/json')
        
    def process(path_file):
        text = ""
        filename = secure_filename(path_file)
        if filename != '':
            conn = sqlite3.connect(os.path.join(current_app.config['DB_PATH'], 'auditree.db'), timeout=10)
            c = conn.cursor()
            c.execute("SELECT text_raw,text_network_object,benford_object FROM corpus WHERE filename = ?", [filename])
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
                    
                if list(data[0])[2] is None:
                    p=re.compile(r"^(19|20)\d{2}$")
                    res=[]
                    reader = PyPDF2.PdfFileReader(file_path)
                    for page in reader.pages:  
                        arr = re.findall(r"[\d.]*\d+", page.extractText())
                        for a in arr:
                            a=str(a)
                            if p.findall(a):
                                continue
                            a=a.replace(".","")
                            if len(a)>17:
                                a=a[:16]
                            res.append(a)
                    df = pd.DataFrame(list(res),columns=['nilai']) 
                    fld=bf.first_digits(data=df['nilai'].astype(np.float),digs=1,decimals=8,confidence=95)
                    result_json = fld.to_json(orient="split")
                    result = json.loads(result_json)
                    c.execute("UPDATE corpus SET benford_object = ? WHERE filename = ?",[result_json,filename])
            conn.commit()
            conn.close()
        return redirect(url_for('home_blueprint.index_filename',filename=filename))