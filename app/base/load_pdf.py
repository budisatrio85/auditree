import tabula
import os
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
base_path = "D:/SOURCE/AudiTREE/document"
file_path = os.path.join(base_path + "/" + "laporan-keuangan-2018.pdf")
print(file_path)
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
        print("PAR")
        print(x)
        if isinstance(x, str) and len(x) > 64:
            temp_arr = [page_no, x]
            print("ARR")
            print(temp_arr)
            df_data.append(temp_arr)
        input()
        
    df =pd.DataFrame(df_data, columns=['Page', 'Content'])
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
        
    df =pd.DataFrame(df_data, columns=['Page', 'Content'])
    df['Content'] = df['Content'].astype('string')

    return df

def get_text_from_pdf(file_path, format_result='paragraph'):
    output, output_2 = convert_pdf_to_string(file_path)
    df_frames = []
    for x in range(0, len(output.index)):
        text = (output.loc[output['Page'] == x+1, 'Content']).to_string(index=False)
        if format_result == 'paragraph' :
            df = extract_paragraph(x+1, text)
        else: 
            df = extract_sentences(x+1, text)

        df_frames.append(df)

    data_df = pd.concat(df_frames)
    json_data = data_df.to_json(orient="split")
    print(json_data)

get_text_from_pdf(file_path)
# get_text_from_pdf(file_path, 'sentences')
