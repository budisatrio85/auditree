import tabula
import re
import benford as bf
from matplotlib import pyplot as plt
import pandas as pd
import os
from fuzzywuzzy import process
from glob import glob
import PyPDF2



class victorinox(object):
    def victorinox(self):
        return

    def convert_pdf_tables_to_csv(self,
                                  fp = r"../../../corpus/bpkhackathon/telkom/telkom_2019.pdf",
                                  fp_csv = r"../../../corpus/bpkhackathon/telkom/table_telkom_2019.csv"):
        tabula.convert_into(fp, fp_csv,output_format="csv", pages="all")
        print("DONE")
        return

    def extract_currency_from_text(self,
                                   fp_csv=r"../../../corpus/bpkhackathon/telkom/table_telkom_2019.csv",
                                   result_csv=r"../../../corpus/bpkhackathon/telkom/telkom_numbers.csv"):
        res=[]
        p=re.compile(r"^(19|20)\d{2}$")
        with open (fp_csv,"r") as fi:
            with open(result_csv,"w") as fo:
                fo.write("val")
                fo.write("\n")
                for line in fi:
                    arr = re.findall(r"[\d.]*\d+", line)
                    for a in arr:
                        a=str(a)
                        if p.findall(a):
                            continue
                        a=a.replace(".","")
                        if len(a)>17:
                            a=a[:16]
                        #fo.write(a[:5])
                        fo.write(a)
                        fo.write("\n")
                        res.append(a)
            return(res)


    def extract_all_currency_from_pdf(self,
                                  fp="/a.csv",
                                  result="/b.txt"
                                  ):
        pdfFileObject = open(fp, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        p = re.compile(r"^(19|20)\d{2}$")
        with open(result,"w") as fo:
            for i in range(count):
                page = pdfReader.getPage(i)
                lines=str(page.extractText()).split("\n")
                for line in lines:
                    arr = re.findall(r"[\d.,]*\d+", line)
                    for number in arr:
                        a = str(number)
                        if p.findall(a):  # skip year
                            continue
                        a = a.replace(".", "").replace(",", "")  # remove dots & comma in numeric
                        if len(a) > 17:
                            a = a[:16]
                        fo.write(a)
                        fo.write("\n")


    def extract_individual_currency(self,
                                    source_folder="/a",
                                    dest_folder="/c"):
        pdfs=glob(os.path.join(source_folder,"**/*.pdf"),recursive=True)
        totalpdf=len(pdfs)
        count=1
        for path, directories, files in os.walk(source_folder):
            if files:
                for file in files:
                    if str(file).endswith(".pdf"):
                        try:
                            pdffile = os.path.join(path, file)
                            print("############## {} / {} : {}".format(count,totalpdf,pdffile))
                            temp_dst = path[len(source_folder) + 1:]
                            temp_dir = os.path.join(dest_folder, temp_dst)
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)
                            fn = "all_num_" + file + ".txt"
                            all_currency_fp= os.path.join(temp_dir, fn)
                            self.extract_all_currency_from_pdf(fp=pdffile,result=all_currency_fp)
                            count+=1
                        except Exception as e:
                            print("Error on file {} : {} ".format(pdffile,str(e)))
                            continue
                            count+=1


    def plot_benford(self,
                     df,
                     digs=1,
                     decimals=8,
                     confidence=0.95):
        fld=bf.first_digits(data=df,digs=digs,decimals=decimals,confidence=confidence)
        plt.show()
        return fld

    def extract_population_currency(self,
                                 item_dictionary="/a.csv",
                                   src_folder=r"../../../corpus/bpkhackathon/telkom/",
                                 raw_folder=r"/a",
                                   dest_folder=r"../../../corpus/bpkhackathon/telkom/numbers"):
        res = []
        item_master=[]
        pdffiles=glob(os.path.join(src_folder, "**/*.pdf"),recursive=True)
        totalpdf=len(pdffiles)
        counter=1
        error_path=os.path.join(dest_folder,"file_error.txt")
        with open(item_dictionary,"r") as fi:
            item_master=[item.replace("\n","") for item in fi.readlines()]
        for path, directories, files in os.walk(src_folder):
            if files:
                for file in files:
                    grabbed_keys={}
                    try:
                        if file.lower().endswith(".pdf"):
                            print("################ {} / {} : {} ###############".format(counter,totalpdf,os.path.join(path,file)))
                            pdf_fp=os.path.join(path, file)
                            temp_dst = path[len(src_folder) + 1:]
                            temp_dir = os.path.join(raw_folder, temp_dst)
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)
                            fn="table_"+file+".csv"
                            table_csv=os.path.join(temp_dir,fn)
                            self.convert_pdf_tables_to_csv(fp=pdf_fp,
                                                           fp_csv=table_csv)
                            res = []
                            p = re.compile(r"^(19|20)\d{2}$")
                            fn2="currency_"+file+".csv"
                            currency_csv=os.path.join(temp_dir,fn2)
                            prev_label=""
                            with open(table_csv, "r") as fi:
                                with open(currency_csv, "w") as fo:
                                    # fo.write("val")
                                    # fo.write("\n")
                                    for line in fi:
                                        try:
                                            item_label = line.split(",")[0]
                                            arr = re.findall(r"[\d.,]*\d+", line)
                                            if len(arr)>=2:
                                                if item_label=="\"\"":      #use previosu linle first words if label is ""
                                                    item_label=prev_label
                                                else:
                                                    ps = process.extractOne(item_label,
                                                                            item_master,
                                                                            score_cutoff=50)
                                                    if ps:
                                                        target_item=str(ps[0]).split(";")[-1] +".txt" #get highest similar item: 1,2,3,4,...9
                                                        if target_item not in grabbed_keys:
                                                            target_fp=os.path.join(dest_folder,target_item)
                                                            with open(target_fp,"a+") as item_fo:
                                                                for a in arr:
                                                                    a = str(a)
                                                                    if p.findall(a):   #skip year
                                                                        continue
                                                                    a = a.replace(".", "").replace(",","") #remove dots & comma in numeric
                                                                    if len(a) > 17:
                                                                        a = a[:16]
                                                                    # fo.write(a[:5])
                                                                    fo.write(a)
                                                                    fo.write("\n")
                                                                    item_fo.write(a)
                                                                    item_fo.write("\n")
                                                                    res.append(a)
                                                            grabbed_keys[target_item]=True
                                                        else:
                                                            pass
                                            prev_label=item_label
                                        except Exception as e:
                                            print("Error file {} line {}. Error:{}".format(table_csv,line,str(e)))
                                            counter+=1
                                            with open(error_path,"a+") as er:
                                                er.write(table_csv)
                                                er.write("\n")
                                            continue
                            counter+=1
                    except Exception as e:
                        print("Error file {} line {}. Error:{}".format(file, line, str(e)))
                        counter+=1
                        with open(error_path, "a+") as er:
                            er.write(table_csv)
                            er.write("\n")
                        continue
        return (res)


    def extract_population_currency_by_year(self,
                                 item_dictionary="/a.csv",
                                   src_folder=r"../../../corpus/bpkhackathon/telkom/",
                                 raw_folder=r"/a",
                                   dest_folder=r"../../../corpus/bpkhackathon/telkom/numbers",
                                            year=2019):
        res = []
        item_master=[]
        pdffiles=glob(os.path.join(src_folder, "**/*.pdf"),recursive=True)
        totalpdf=len(pdffiles)
        counter=1
        error_path=os.path.join(dest_folder,"file_error.txt")
        p1=re.compile("[1-2][0-9]{3}")
        with open(item_dictionary,"r") as fi:
            item_master=[item.replace("\n","") for item in fi.readlines()]
        for path, directories, files in os.walk(src_folder):
            if files:
                for file in files:
                    grabbed_keys={}
                    try:
                        f=file
                        if f.lower().endswith(".pdf"):
                            y=[]
                            y=p1.findall(str(f))
                            if y:
                                if int(y[0])==year:
                                    print(y[0])
                                    print("################ {} / {} : {} ###############".format(counter,totalpdf,os.path.join(path,file)))
                                    pdf_fp=os.path.join(path, file)
                                    temp_dst = path[len(src_folder) + 1:]
                                    temp_dir = os.path.join(raw_folder, temp_dst)
                                    if not os.path.exists(temp_dir):
                                        os.makedirs(temp_dir)
                                    fn="table_"+file+".csv"
                                    table_csv=os.path.join(temp_dir,fn)
                                    self.convert_pdf_tables_to_csv(fp=pdf_fp,
                                                                   fp_csv=table_csv)
                                    res = []
                                    p = re.compile(r"^(19|20)\d{2}$")
                                    fn2="currency_"+file+".csv"
                                    currency_csv=os.path.join(temp_dir,fn2)
                                    prev_label=""
                                    with open(table_csv, "r") as fi:
                                        with open(currency_csv, "w") as fo:
                                            # fo.write("val")
                                            # fo.write("\n")
                                            for line in fi:
                                                try:
                                                    item_label = line.split(",")[0]
                                                    arr = re.findall(r"[\d.,]*\d+", line)
                                                    if len(arr)>=2:
                                                        if item_label=="\"\"":      #use previosu linle first words if label is ""
                                                            item_label=prev_label
                                                        else:
                                                            ps = process.extractOne(item_label,
                                                                                    item_master,
                                                                                    score_cutoff=50)
                                                            if ps:
                                                                target_item=str(ps[0]).split(";")[-1] +".txt" #get highest similar item: 1,2,3,4,...9
                                                                if target_item not in grabbed_keys:
                                                                    target_fp=os.path.join(dest_folder,target_item)
                                                                    with open(target_fp,"a+") as item_fo:
                                                                        for a in arr:
                                                                            a = str(a)
                                                                            if p.findall(a):   #skip year
                                                                                continue
                                                                            a = a.replace(".", "").replace(",","") #remove dots & comma in numeric
                                                                            if len(a) > 17:
                                                                                a = a[:16]
                                                                            # fo.write(a[:5])
                                                                            fo.write(a)
                                                                            fo.write("\n")
                                                                            item_fo.write(a)
                                                                            item_fo.write("\n")
                                                                            res.append(a)
                                                                    grabbed_keys[target_item]=True
                                                                else:
                                                                    pass
                                                    prev_label=item_label
                                                except Exception as e:
                                                    print("Error file {} line {}. Error:{}".format(file,line,str(e)))
                                                    counter+=1
                                                    with open(error_path,"a+") as er:
                                                        er.write(table_csv)
                                                        er.write("\n")
                                                    continue
                                    counter+=1
                    except Exception as e:
                        print("Error file {} line {}. Error:{}".format(file, line, str(e)))
                        counter+=1
                        with open(error_path, "a+") as er:
                            er.write(f)
                            er.write("\n")
                        continue
        return (res)


    def remove_last_2_zero_digits(self,
                             source_folder="/a",
                             dest_fodler="/b"):
        files=glob(os.path.join(source_folder, "**/*.txt"),recursive=True)
        for file in files:
            filename=os.path.split(file)[-1]
            with open(file, "r") as fi:
                fopath=os.path.join(dest_fodler,filename)
                with open(fopath,"w") as fo:
                    for line in fi:
                        line=line.strip()
                        if line.endswith("00"):
                            line=line[:-2]
                        fo.write(line)
                        fo.write("\n")

    def extract_benford(self,currency_path="/a.csv",
                                   digs=1,
                               show_plot=False):
        df=pd.read_csv(currency_path,header=None)

        points= bf.first_digits(data=df.iloc[:,0],
                               digs=digs,
                               show_plot=False)
        violations = bf.first_digits(data=df.iloc[:, 0],
                                 digs=digs,
                                 show_plot=show_plot,
                                 high_Z="pos",
                                 confidence=95)
        return (points,violations)


