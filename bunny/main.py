import base64

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

app = FastAPI()

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
        # contents = await file.read()
        # blob = base64.b64decode(contents)
        # pdf = open('result.pdf', 'wb')
        # pdf.write(blob)
        # pdf.close()
        # return {"filename": file.filename}
        # # f=pdf
        stream = io.BytesIO(pdf)
        # text_io = io.TextIOWrapper(stream)
        # text_io.seek(0, 2)  # seek to end of file; f.seek(0, 2) is legal
        # # text_io.seek(text_io.tell() - 3, os.SEEK_SET)
        # # print(type(stream))
        # pdfFileObject=text_io
        pdfReader = PyPDF2.PdfFileReader(stream)
        count = pdfReader.numPages
        result.status="ok"
        result.graph={}
    except Exception as e:
        print(str(e))
        result.status="errror"
        result.graph={"eerror":str(e)}
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
