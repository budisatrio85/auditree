# import torch
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())

# import re
# import textract
# #read the content of pdf as text
# fp=r"../../corpus/bpkhackathon/telkom/FS Q12018_INDONESIA.pdf"
# text = textract.process(fp)
# #use four space as paragraph delimiter to convert the text into list of paragraphs.
# print (re.split('\s{4,}',text))


# import os
# import re
#
# fp="lapkeu.txt"
# content=""
# pars=[]
# with open(fp,"r") as fi:
#     for line in fi:
#         stripped=line.strip()
#         content+=" "+stripped
#         if line.endswith(".\n"):
#             pars.append(content)
#             content=""
# paragraphs = content.split(".\n+")
# print("NUM OF PAR:",len(paragraphs))
# with open("lapkeu_paragraph.txt","w") as fo:
#     for p in pars:
#         if (p):
#             if(len(p.split())>=3):
#                 p=p.strip()
#
#                 fo.write(p.strip())
#                 fo.write("\n")

# fp="lapkeu_paragraph.txt"
# with open(fp,"r") as fi:
#     for par in fi:
#         print(par)
#         par = str(par).replace("\n", "")
#         if "\n" in par:
#             print(":(")
#         break

# import re
# text="PT Telkom Landmark Jasa pengembangan dan,2012,55,55,,2.056,,2.128,.667\n"
# res=re.findall(r"\d+[.]*\d+", text)
# print(res)


# import re
# teks="1019"
# p=re.compile(r"^(19|20)\d{2}$")
# if p.findall(teks):
#     print("found")


#############################



# import pandas as pd
# import numpy as np
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
#
# d1={1:'jumlah pendapatan negara','2':'belanja barang',3:'Sally Can sing',4:'Dicke bauch',5:'ASet'}
# d2={1:'Jumlah Asset tetap',6:"jumlah pendapatan",'2':'belanja negara 2',3:'jumlah belanja;0',4:'Aset lain-lain 5',5:'jumlah aset'}
#
# df1=pd.DataFrame.from_dict(d1,orient='index')
# df2=pd.DataFrame.from_dict(d2,orient='index')
#
# df1.columns=['Name']
# df2.columns=['Name']
#
# def match(Col1,Col2):
#     overall=[]
#     r=[]
#     for n in Col1:
#         result=[(fuzz.partial_ratio(n, n2),n2)
#                 for n2 in Col2 if fuzz.partial_ratio(n, n2)>50
#                ]
#         if len(result):
#             result.sort()
#             print('result {}'.format(result))
#             print("Best M={}".format(result[-1][1]))
#             overall.append(result[-1][1])
#         else:
#             overall.append(" ")
#         ps = process.extractOne(n, Col2, score_cutoff=76)#scorer=fuzz.token_sort_ratio)
#         if ps:
#             print("{} --> {}".format(n,ps[0]))
#         else:
#             print("{} --> []".format(n))
#         r.append(ps)
#     return (overall,r)
#
# x,y=match(df1.Name,df2.Name)
# print(x)
# print(y)


############################

# from transformers import AutoTokenizer, AutoModel
# import numpy as np
# import torch
# from scipy import spatial
#
# MAX_TOKEN_LENTH=59
# tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
# model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
# a="penerimaan negara. penerimaan negara"
# b="PENDAPATAN NEGARA DAN HIBAH Penerimaan Perpajakan. Pendapatan negara bukan pajak."
# c="BELANJA NEGARA. Belanja Pegawai."
#
# tokena=tokenizer.encode(a,add_special_tokens=True)
# tokenb=tokenizer.encode(b,add_special_tokens=True)
# tokenc=tokenizer.encode(c,add_special_tokens=True)
#
# tokens=[tokena,tokenb,tokenc]
# max_len = 0
# for i in tokens:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])
#
# print(padded)
# attention_mask = np.where(padded != 0, 1, 0)
#
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# features = last_hidden_states[0][:,0,:].numpy()
#
# # for f in features:
# #     print(f)
#
# distab = 1 - spatial.distance.cosine(features[0], features[1])
# distac = 1 - spatial.distance.cosine(features[0], features[2])
# distbc = 1 - spatial.distance.cosine(features[1], features[2])
# print(distab)
# print(distac)
# print(distbc)

# with open("table_lk-2018.csv", "r") as fi:
#     for l in fi:
#         arr=l.split(",")
#         if arr[0]=="\"\"":
#             print(arr)



#
# a={}
# a["1"]=True
# if "2" not in a:
#     print(a)


# import benford as bf
# from victorinox import victorinox
# import pandas as pd
# from matplotlib import pyplot as plt
# from benford import Benford as bf2
# tool=victorinox()
# fp_currency_csv=r"corpus/aggregate_clean/3.txt"#dki/numeric_lkpd-2015.csv"
# df=pd.read_csv(fp_currency_csv,header=None)
# suspect=bf.first_digits(data=df.iloc[:,0],digs=1,decimals=0)
# plt.show()
# # for idx,row in suspect.iterrows():
# #     print(row)
# suspect=bf.last_two_digits(data=df.iloc[:,0])
# plt.show()

# from victorinox import victorinox
# tool=victorinox()
# tool.create_last_2_digits("bunny/population_2019","bunny/aggregate_clean")


# import PyPDF2
# pdfFileObject = open(r'corpus/source/kemenkeu/lk-2018.pdf', 'rb')
# pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
# count = pdfReader.numPages
# for i in range(count):
#     page = pdfReader.getPage(i)
#     lines=str(page.extractText()).split("\n")
#     for line in lines:
#         print(line)
#         print("##############################")


# from victorinox import victorinox
# tool=victorinox()
# import matplotlib.pyplot as plt
# # tool.extract_all_currency_from_pdf("bunny/kemenkeu/lk-2008.pdf","bunny/kemenkeu/all_num_lk-2008.txt")
#
# # tool.extract_individual_currency("bunny/kemenkeu","bunny/all_currency")
#
# print(tool.extract_bendorf(r"corpus/individu/kemenagra/all_num_lk-2019.pdf.txt",digs=1,show_plot=True))
# plt.show()
# print(tool.extract_bendorf(r"corpus/population_2019/4.txt",digs=1,show_plot=True))
# plt.show()
# print(tool.extract_bendorf(r"corpus/individu/kemenagra/all_num_lk-2019.pdf.txt",digs=2,show_plot=True))
# plt.show()


# import re
# import os
# from collections import Counter
# p=re.compile("\d+")
# print(p.findall("all_num_lk-2019.pdf.csv"))
# arr=[]
# for path, fodlers, files in os.walk("corpus/source"):
#     if files:
#         for file in files:
#             if file.endswith(".pdf"):
#                 y=p.findall(file)
#                 if y:
#                     arr.append(y[0])
# print(Counter(arr))

# from victorinox import victorinox
# tool=victorinox()
# tool.extract_population_currency_by_year(item_dictionary="report_items.csv",
#                                    src_folder=r"corpus/source",
#                                  raw_folder=r"bunny/history",
#                                    dest_folder=r"bunny/population_2019",
#                                          year=2019)

#
# import re
# import os
# from collections import Counter
# p=re.compile("\d+")
# print(p.findall("lk-2016.pdf.csv"))

# from victorinox import victorinox
# tool=victorinox()
# x,y=tool.extract_benford(r"corpus/population/4.txt",digs=1)
# print(x)
# print(y.head())


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





pd.set_option('display.max_colwidth', None)
SAVED_FILES = 'indobert_files'
BASE_PATH = "financeReport"
CHARACTER_THRESHOLD = 350
FILE_PATH = os.path.join(BASE_PATH + "/" + "laporan-keuangan-2018.pdf")
ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')


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

sentences="Catatan atas Laporan Keuangan (CaLK) menyajikan informasi tentang penjelasan atau daftar terinci atau analisis atas nilai " \
         "suatu pos yang disajikan dalam Laporan " \
         "Realisasi Anggaran, Neraca, Laporan Operasional, dan Laporan Perubahan Ekuitas. Termasuk pula dalam CaLK " \
         "adalah penyajian informasi yang diharuskan dan dianjurkan oleh Standar Akuntansi " \
         "Pemerintahan serta pengungkapan-pengungkapan lainnya yang diperlukan untuk penyajian " \
         "yang wajar atas laporan keuangan. Dalam penyajian Laporan Realisasi Anggaran untuk periode yang berakhir sampai dengan tanggal 31 Desember 2018 " \
          " disusun dan disajikan berdasarkan basis kas. Sedangkan Neraca, Laporan Operasional, dan Laporan Perubahan Ekuitas sampai dengan 31 Desember 2018 " \
          "disusun dan disajikan dengan menggunakan basis akrual."

fp=r"../auditree.local/corpus/source/kemenkeu/lk-2018.pdf"
pdfFileObject = open(fp, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
count = pdfReader.numPages
p = re.compile(r"^(19|20)\d{2}$")
# with open(result,"w") as fo:
total_nouns=[]
data=[]
for i in range(count):
    page = pdfReader.getPage(i)
    txt=page.extractText()
    lines=str(txt).split("\n")
    sentences=""
    for line in lines:
        arr = re.findall(r"[a-zA-Z]+", line)
        sentences=sentences+" "+ " ".join([w for w in arr])
    paragraph_nouns=[]
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
                temp_noun=""
                sentence_nouns=[]
                prev_pos=""
                for text, pos in hasil[0]:
                    # print("{}:{}".format(text,pos))
                    if (pos == "NN" or pos =="NNP") and (prev_pos == "NN" or prev_pos =="NNP") :
                        if len(temp_noun.split())<2:
                            temp_noun=temp_noun+" "+text
                            temp_noun=str(temp_noun).lower()
                    elif (pos != "NN" or pos !="NNP") and (prev_pos == "NN" or prev_pos =="NNP"):
                        if temp_noun:
                            temp_noun=remove_punctuation(temp_noun)
                            total_nouns.append(temp_noun)
                            sentence_nouns.append(temp_noun.strip())
                            if len(sentence_nouns)==2:
                                paragraph_nouns.append(sentence_nouns)
                                data.append(sentence_nouns)
                                sentence_nouns=[]
                        temp_noun=""
                    prev_pos=pos
                    prev_text=text
                if sentence_nouns:
                    if len(sentence_nouns)==2:
                        paragraph_nouns.append(sentence_nouns)
            except Exception as e:
                print("Eror: line {}".format(s))
                continue
    print(paragraph_nouns)
c=Counter(total_nouns)
total_words=len(total_nouns)
node=[]
T=30
minor=[]
for x in c:
    key = x
    value = c[key]
    if value>T:
        d={
          "id": key,
          "marker": {
            "radius": math.ceil((value/total_words) * 1000)
          },
          "color": "#%06x" % random.randint(0, 0xFFFFFF)
        }
        node.append(d)
    else:
        minor.append(key)
print(c)
print(minor)
ret={"data":[d for d in data if (d[0] not in minor) or (d[1] not in minor) ],
     "node":node,
     "min_weight": 19}
print(ret)

# a=open(r"tfidf_id.csv","r")
# print(type(a))