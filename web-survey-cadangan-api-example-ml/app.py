#!/usr/bin/env python
# encoding: utf-8

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

#untuk ML
from flask import Flask, render_template, request
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,recall_score
import pandas as pd
import string
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np 

app = Flask(__name__)
CORS(app)

#open model

########### BAGIAN IMPORT FILE ####################
# import kamus slang

slang_words = pd.read_csv('dict/slang_words.csv', header=None)
slang_words = slang_words.rename(columns={0: 'original', 1: 'replacement'}) 
#membuat jadi dictionary
slang_words_map = dict(zip(slang_words['original'], slang_words['replacement']))

#import kamus kata-kata positif

positive_dict = pd.read_csv('dict/kamus_positif.csv')
dict_positive = positive_dict['kata positif'].to_dict()

# load saved model algoritma naive 

file_model_naive = open('naive_classifier.pkl', 'rb')
model_naive = pickle.load(file_model_naive)
file_model_naive.close()

#load save model fit tfidf vocab

file_model_tfidf_vocab = open('tfidf_vocab30ns.pkl', 'rb')
vocabularies = pickle.load(file_model_tfidf_vocab)
file_model_tfidf_vocab.close()

file_model_tfidf_idf = open('tfidf_idf30ns.pkl', 'rb')
idf_result = pickle.load(file_model_tfidf_idf)
file_model_tfidf_idf.close()

text_answer = []

###############################################


def POST(data_req):

    url = "http://127.0.0.1:8000/api/survey"
    data = data_req.to_dict(flat=False)

    data_ans ={
        "question_1":data['question_1'][0],
        "question_2":data['question_2'][0],
        "question_3":data['question_3'][0],
        "question_4":data['question_4'][0],
        "question_5":data['question_5'][0],
    }
    

    for i in dict.values(data_ans) :
        text_answer.append(i)
    
    preproces_data = preprocessing(text_answer)
             
    #TFIDF
        
    final_answer_new=transform(preproces_data.tolist(), vocabularies, idf_result)
    final_tfidf_answer_new = final_answer_new.toarray()

    label_predicts = model_naive.predict(final_tfidf_answer_new) 
    print(label_predicts)

    data_obj = {
        "name":data['name'][0],
        "email":data['email'][0],
        "age":data['age'][0],
        "prodi":data['prodi'][0],
        "semester":data['semester'][0],
        "gender":data['gender'][0],
        "question_1":data['question_1'][0],
        "question_2":data['question_2'][0],
        "question_3":data['question_3'][0],
        "question_4":data['question_4'][0],
        "question_5":data['question_5'][0],
        "hasil_klasifikasi1": label_predicts[0],
        "hasil_klasifikasi2": label_predicts[1],                
        "hasil_klasifikasi3": label_predicts[2],    
        "hasil_klasifikasi4": label_predicts[3],    
        "hasil_klasifikasi5": label_predicts[4],             
    }

    response = requests.post(url, data=data_obj)
    print(response.text)
    
@app.route('/api/survey', methods = ['POST', 'DELETE'])
def user():
    
    if request.method == 'POST':
        data_request = request.form
        POST(data_request)
        
        return 'Data Berhasil Ditambahkan !'

    else:
        print("Error")


@app.route('/', methods = ['GET'])
def index():
    
    if request.method == 'GET':
        return 'Api Flask for Machine Learning.'
    else:
        print("Error")


#ML

def remove_emoji_utf8(edom_respond):
    
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    
    edom_respond = re.sub(emoj, '', edom_respond)
                  
    """ MENGHILANGKAN UNICODE DAN EMOJI DARI UTF-8 """
    edom_respond = edom_respond.encode('utf-8', errors='ignore').decode('unicode-escape', errors='ignore')
    
    edom_respond = edom_respond.encode("ascii", 'ignore')
    edom_respond = edom_respond.decode("ascii")
    
    return edom_respond

def remove_karakter(edom_respond):

    edom_respond = edom_respond.lower() #RUBAH MENJADI HURUF KECIL SEMUA
    edom_respond = re.sub(r'#([^\s]+)', '  ', edom_respond) #MENGHILANGKAN HASHTAG
    edom_respond = re.sub('(www\.[^\s]+)|(https?://[^\s]+)','  ',edom_respond) #MENGHILANGKAN URL
    edom_respond = re.sub(r"\\n", ' ', edom_respond) #hilangin enter

    edom_respond = edom_respond.translate(str.maketrans("  ","  ",string.punctuation)) #hilangin tanda baca lainnya
    
    edom_respond = ''.join([i for i in edom_respond if not i.isdigit()]) #MENGHILANGKAN ANGKA
    edom_respond = re.sub(r'\s+', ' ', edom_respond).strip() #menghilangkan spasi berlebih

    edom_respond = ' '.join( [w for w in edom_respond.split() if len(w)>1]) #MENGHILANGKAN HURUF SATUAN ex 'aaku' jadi aku
    edom_respond = re.sub(r'(.+?)\1+', r'\1', edom_respond) #MENGHILANGKAN HURUF SATUAN ex 'aku a' a akan ilang
    edom_respond = re.sub(r'\s+', ' ', edom_respond).strip() #menghilangkan spasi berlebih

    
    edom_respond = ' '.join([slang_words_map[word] if word in slang_words_map else word for word in edom_respond.split(' ')]) #MERUBAH BAHASA SLANG
    edom_respond = re.sub(r'\s+', ' ', edom_respond).strip() #menghilangkan spasi berlebih

    return edom_respond

factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

def stemming(edom_respond):
    
    edom_respond = edom_respond.strip() 
    edom_respond = stemmer.stem(edom_respond)
    
    return edom_respond

#BAGIAN TFIDF

def IDF(corpus, unique_words):
    idf_dict ={}
    N=len(corpus)
    for i in unique_words:
        count =0
        for sent in corpus:
            if i in sent.split():
                count=count+1
            idf_dict[i]=(1+np.log10((1+N)/(count+1)))
    return idf_dict

def fit(whole_data):

    unique_words = set()
    if isinstance(whole_data, (list,)):
        for x in whole_data:
            for y in x.split():
                if len(y)<2:
                    continue
            unique_words.add(y)
    unique_words = sorted(list(unique_words))
    vocab = {j:i for i,j in enumerate(unique_words)}
    Idf_values_of_all_unique_words=IDF(whole_data,unique_words)

    return vocab, Idf_values_of_all_unique_words

def transform(dataset,vocabulary,idf_values):
    sparse_matrix= csr_matrix( (len(dataset), len(vocabulary)), dtype=np.float64)
    for row  in range(0,len(dataset)):
        number_of_words_in_sentence=Counter(dataset[row].split())
        for word in dataset[row].split():
            if word in list(vocabulary.keys()):
                if word in list(dict_positive.values()):
                    tf_idf_value=((number_of_words_in_sentence[word]/len(dataset[row].split()))*(idf_values[word]))+1
                else:
                    tf_idf_value=((number_of_words_in_sentence[word]/len(dataset[row].split()))*(idf_values[word]))

                sparse_matrix[row,vocabulary[word]]=tf_idf_value
    output =normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
    return output

def preprocessing(texts):
    
    df = pd.DataFrame(texts,columns =['answer'])
    
    df['remove_emoji'] = df['answer'].apply(lambda text: remove_emoji_utf8(text))
    df['cleaned_answer'] = df['remove_emoji'].apply(lambda text: remove_karakter(text))
    df['stem_answer'] = df['cleaned_answer'].apply(lambda text: stemming(text))
        
    #return df['cleaned_answer'] 
    return df['stem_answer'] 

if __name__ == '__main__':
    app.run()