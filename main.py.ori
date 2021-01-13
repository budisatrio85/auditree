from fastapi.responses import JSONResponse
from fastapi import FastAPI, File,UploadFile
from starlette.requests import Request
import uvicorn
from pydantic import BaseModel
import pickle
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import io
from victorinox import victorinox
from glob import glob
import os
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np #Operasi Matematika dan linear aljebra
import pandas as pd #data processing
import matplotlib.pyplot as plt #Visualisasi data
import string
import nltk
nltk.download("punkt")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
from scipy import spatial
import math
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import heapq
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import numpy as np
from nltk.tag import CRFTagger
import string
import nltk
nltk.download("punkt")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas as pd
import re
import PyPDF2
from collections import Counter
import math
import random

ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')


tool=victorinox()
population1_dict={}
population2_dict={}
population_root_path=r"corpus/population"
population_files=glob(os.path.join(population_root_path, "**/*.txt"),recursive=True)
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
default_stopwords = StopWordRemoverFactory().get_stop_words()
additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
id_stopword = StopWordRemover(dictionary)
en_stopword = set(stopwords.words('english'))
en_stemmer = PorterStemmer()

def remove_numbers(text):
  words=tokenizer.tokenize(text)
  return " ".join(words)

def remove_punctuation(text):
  words = text.split()
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in words]
  return " ".join(stripped)

def stem_text(text):
  return stemmer.stem(text)

def remove_stopwords(text):
  return id_stopword.remove(text)

def remove_english_stopwords(text):
  if text:
    return " ".join([token for token in text.split() if token not in en_stopword])

def stem_english_text(text):
  return " ".join([en_stemmer.stem(word) for word in text.split()])

def remove_single_char(text):
  return " ".join([ x for x in text.split() if len(x)>1])


for f in population_files:
    id=os.path.split(f)[-1].replace(".txt","")
    (points,violations)=tool.extract_benford(currency_path=f,
                         digs=1)
    population1_dict[id]=(points,violations)
    (points2, violations2) = tool.extract_benford(currency_path=f,
                                                digs=2)
    population2_dict[id] = (points2, violations2)

individu1_dict={}
individu2_dict={}
individu_root_path=r"corpus/individu"
individu_files=glob(os.path.join(individu_root_path, "**/*.txt"),recursive=True)

report_list=[]
pp=re.compile("\d+")
for f in individu_files:
    try:
        folder,fn=os.path.split(f)
        company=str(folder).split("/")[-1]
        year=pp.findall(fn)[0]
        id=company+"_"+year
        report_list.append(id)
        (points,violations)=tool.extract_benford(currency_path=f,
                             digs=1)
        individu1_dict[id]=(points,violations)
        (points2, violations2) = tool.extract_benford(currency_path=f,
                                                    digs=2)
        individu2_dict[id] = (points2, violations2)
    except Exception as e:
        print("Error file {}:{}".format(f,str(e)))
        continue

df = pd.read_csv(r'report_items2.csv',sep=";")
report_cols=["id","report"]
report_items=df[report_cols]
tfidf_model_path="tfidf_model.pkl"
vector_id_path="tfidf_id.csv"
knn_model_path=r'knn_model.pkl'
knn_index_path=r'knn_index.pkl'
report_item_master_csv=r'report_item_master.csv'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
default_stopwords = StopWordRemoverFactory().get_stop_words()
additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
id_stopword = StopWordRemover(dictionary)
en_stopword = set(stopwords.words('english'))
en_stemmer = PorterStemmer()
def remove_numbers(text):
  words=tokenizer.tokenize(text)
  return " ".join(words)

def remove_punctuation(text):
  words = text.split()
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in words]
  return " ".join(stripped)

def stem_text(text):
  return stemmer.stem(text)

def remove_stopwords(text):
  return id_stopword.remove(text)

def remove_english_stopwords(text):
  if text:
    return " ".join([token for token in text.split() if token not in en_stopword])

def stem_english_text(text):
  return " ".join([en_stemmer.stem(word) for word in text.split()])

def remove_single_char(text):
  return " ".join([ x for x in text.split() if len(x)>1])

print("LOADING CLASSIFIER MODEL")
with open(tfidf_model_path, "rb") as fo:
    fitted_tfidf_vectorizer=pickle.load(fo)
    print("LOADED tfidf model2 from ", tfidf_model_path)

with open(knn_index_path,"rb") as fi:
  knn_index=pickle.load(fi)
  print("LOADED KNN index from ", knn_index_path)

with open(knn_model_path,"rb") as fi:
  knn_model=pickle.load(fi)
  print("LOADED KNN model from ", knn_model_path)


report_item_master=pd.read_csv(report_item_master_csv,header=0, sep=";")

class BenfordModel(BaseModel):
    status: str
    points: dict
    violations: dict

class ListModel(BaseModel):
    status: str
    report:list


# m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': 123})
class Benford_Item(BaseModel):
    status: str
    val: dict


app = FastAPI()
@app.get("/get_report_list")
async def get_report_list(request: Request):#,
                 #pdf: bytes = File(...)):

    init_result = {
        'status': 'ok',
        "report": []
    }
    result = ListModel(**init_result)
    result.status="ok"
    result.report=report_list
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/population_benford")
async def population_benford(request: Request):#,
                 #pdf: bytes = File(...)):

    init_result = {
        'status': 'error',
        "val": {}
    }
    result = Benford_Item(**init_result)
    if request.method == "POST":
        try:
            form = await request.form()
            id=form["id"]
            digits = int(form["digits"])
            if digits==1:
                if id in population1_dict:
                    (points,violations)=population1_dict[id]
                    result=BenfordModel(status="ok",
                                        points={"expected":points.Expected,"found":points.Found},
                                        violations={"expected":violations.Expected,"found":violations.Found})
                else:
                    result.status="not-found"
                    result.val="id not found"
            else:
                if id in population2_dict:
                    (points,violations)=population2_dict[id]
                    result = BenfordModel(status="ok",
                                          points={"expected": points.Expected,
                                                  "found": points.Found},
                                          violations={"expected": violations.Expected,
                                                      "found": violations.Found})

                else:
                    result.status="not-found"
                    result.val="id not found"
            # stream = io.BytesIO(pdf)
        except Exception as e:
            result.status="error"
            result.val=str(e)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/individu_benford")
async def individu_benford(request: Request):#,
                 #pdf: bytes = File(...)):
    init_result = {
         'status': 'error',
         "val": {}
     }
    result = Benford_Item(**init_result)
    if request.method == "POST":
        try:
            form = await request.form()
            id=form["id"]
            digits = int(form["digits"])
            if digits==1:
                if id in individu1_dict:
                    (points,violations)=individu1_dict[id]
                    result=BenfordModel(status="ok",
                                        points={"expected":points.Expected,"found":points.Found},
                                        violations={"expected":violations.Expected,"found":violations.Found})
                else:
                    result.status="not-found"
                    result.val="id not found"
            else:
                if id in individu2_dict:
                    (points,violations)=individu2_dict[id]
                    result = BenfordModel(status="ok",
                                          points={"expected": points.Expected,
                                                  "found": points.Found},
                                          violations={"expected": violations.Expected,
                                                      "found": violations.Found})

                else:
                    result.status="not-found"
                    result.val="id not found"
            # stream = io.BytesIO(pdf)
        except Exception as e:
            result.status="error"
            result.val=str(e)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class TopicModel(BaseModel):
    status: str
    code: int
    val:str

@app.post("/classify_topic")
async def verify(request: Request):
    init_result = {
        'status': 'error',
        "code":-1,
        "val": "unknown"
    }
    result = TopicModel(**init_result)
    form = await request.form()
    query = form["query"]
    sentence = query
    sentence = str(sentence).lower()
    sentence = remove_numbers(sentence)
    sentence = remove_punctuation(sentence)
    sentence = remove_stopwords(sentence)
    sentence = stem_text(sentence)
    sentence = remove_english_stopwords(sentence)
    sentence = stem_english_text(sentence)
    sentence = remove_single_char(sentence)
    densematrix = fitted_tfidf_vectorizer.transform([sentence])
    skillvecs = densematrix.toarray()
    vector = np.array(skillvecs[0]).astype('float32')
    vector = np.expand_dims(vector, 0)
    (distances, indices) = knn_model.kneighbors(vector, n_neighbors=5)
    indices = indices.tolist()
    print(indices)
    res = [knn_index[x] for x in indices[0]]
    for id in res:
        result.status="ok"
        result.code=int(id)
        result.val=report_item_master.loc[report_item_master.id == id,["report"]]
        break
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/test")
async def verify(request: Request):
    init_result = {
        'status': 'error',
        "val": "not registered"
    }
    result = Benford_Item(**init_result)
    result.status="ok"
    result.val="jos"
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class GraphModel(BaseModel):
    status: str
    graph:dict
@app.post("/draw_network_graph")
async def draw_network_graph(request: Request,
                 pdf: bytes = File(...)):
    init_result = {
        'status': 'error',
        "graph": {}
    }
    try:
        result = GraphModel(**init_result)
        stream = io.BytesIO(pdf)
        # text_io = io.TextIOWrapper(pdf)
        # text_io.seek(0, os.SEEK_END)  # seek to end of file; f.seek(0, 2) is legal
        # text_io.seek(text_io.tell() - 3, os.SEEK_SET)
        # print(type(stream))
        # pdfFileObject=text_io
        pdfReader = PyPDF2.PdfFileReader(stream)
        count = pdfReader.numPages
        total_nouns = []
        data = []
        for i in range(count):
            page = pdfReader.getPage(i)
            txt = page.extractText()
            lines = str(txt).split("\n")
            sentences = ""
            for line in lines:
                arr = re.findall(r"[a-zA-Z]+", line)
                sentences = sentences + " " + " ".join([w for w in arr])
            paragraph_nouns = []
            if sentences.strip():
                for s in sentences.split("."):
                    try:
                        # s = remove_numbers(s)
                        # s = remove_punctuation(s)
                        # s = remove_stopwords(s)
                        # s = remove_english_stopwords(s)
                        s = remove_single_char(s)
                        # s = stem_text(s)
                        # s = stem_english_text(s)
                        hasil = ct.tag_sents([s.split()])
                        temp_noun = ""
                        sentence_nouns = []
                        prev_pos = ""
                        for text, pos in hasil[0]:
                            # print("{}:{}".format(text,pos))
                            if (pos == "NN" or pos == "NNP") and (prev_pos == "NN" or prev_pos == "NNP"):
                                if len(temp_noun.split()) < 2:
                                    temp_noun = temp_noun + " " + text
                                    temp_noun = str(temp_noun).lower()
                            elif (pos != "NN" or pos != "NNP") and (prev_pos == "NN" or prev_pos == "NNP"):
                                if temp_noun:
                                    temp_noun = remove_punctuation(temp_noun)
                                    total_nouns.append(temp_noun)
                                    sentence_nouns.append(temp_noun.strip())
                                    if len(sentence_nouns) == 2:
                                        paragraph_nouns.append(sentence_nouns)
                                        data.append(sentence_nouns)
                                        sentence_nouns = []
                                temp_noun = ""
                            prev_pos = pos
                            # prev_text = text
                        if sentence_nouns:
                            if len(sentence_nouns) == 2:
                                paragraph_nouns.append(sentence_nouns)
                    except Exception as e:
                        print("Eror: line {}".format(s))
                        continue
            # print(paragraph_nouns)
        c = Counter(total_nouns)
        total_words = len(total_nouns)
        node = []
        T = 5
        minor = []
        for x in c:
            key = x
            value = c[key]
            if value > T:
                d = {
                    "id": key,
                    "marker": {
                        "radius": math.ceil((value / total_words) * 1000)
                    },
                    "color": "#%06x" % random.randint(0, 0xFFFFFF)
                }
                node.append(d)
            else:
                minor.append(key)
        # print(c)
        # print(minor)
        ret = {"data": [d for d in data if (d[0] not in minor) or (d[1] not in minor)],
               "node": node,
               "min_weight": 19}
        result.status="ok"
        result.graph=ret
    except Exception as e:
        result.status="errror"
        result.graph={"eerror":str(e)}
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)



