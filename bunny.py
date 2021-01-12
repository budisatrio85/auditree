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


x="/a/b/c"
print(x.split("/")[-1])