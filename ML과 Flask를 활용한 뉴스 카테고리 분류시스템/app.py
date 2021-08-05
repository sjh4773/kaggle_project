from flask import Flask, render_template, url_for, request
from sklearn.datasets import load_files
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
import konlpy
import nltk
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    def clean_korean_documents(documents):
        #텍스트 정제 (HTML 태그 제거)
        for i, document in enumerate(documents):
            document = BeautifulSoup(document, 'html.parser').text 
            documents[i] = document

        #텍스트 정제 (특수기호 제거)
        for i, document in enumerate(documents):
            document = re.sub(r'[^ ㄱ-ㅣ가-힣]', '', document) #특수기호 제거, 정규 표현식
            documents[i] = document

        #텍스트 정제 (형태소 분석)
        for i, document in enumerate(documents):
            okt = konlpy.tag.Okt()
            clean_words = []
            for word in okt.pos(document, stem=True): #어간 추출
                if word[1] in ['Noun', 'Verb', 'Adjective']: #명사, 동사, 형용사
                    clean_words.append(word[0])
            document = ' '.join(clean_words)
            documents[i] = document

        #텍스트 정제 (불용어 제거)
        df = pd.read_csv('koreanstopwords.txt', header=None)
        df[0] = df[0].apply(lambda x: x.strip())
        stopwords = df[0].to_numpy()
        nltk.download('punkt')
        for i, document in enumerate(documents):
            clean_words = [] 
            for word in nltk.tokenize.word_tokenize(document): 
                if word not in stopwords: #불용어 제거
                    clean_words.append(word)
            documents[i] = ' '.join(clean_words)  

        return documents

    ##########데이터 로드

    naver_news = load_files('C:/Users/SeoJiHun/Desktop/flask_news/newsData', shuffle=True)

    labels = ['정치', '경제', '사회', '생활/문화', '세계', '기술/IT', '연예', '스포츠']

    ##########데이터 분석

    ##########데이터 전처리

    x_data = naver_news.data
    y_data = naver_news.target
    x_data = x_data[:50] #데이터를 50개로 제한
    y_data = y_data[:50]

    print(x_data[0]) #b'\xed\x83\x9c\xec\x96\x91\xea\xb4\x91\xec\x9c\xbc\xeb\xa1\x9c \xec\xb6\xa9\xec\xa0\x84\xeb\x90\x98\xeb\x8a\x94 \xe2\x80\x98\xec\x95\x84\xec\x9d\xb4\xed\x8f\xb0X ...'
    print(y_data[0]) #5

    x_data = [x.decode('utf-8') for x in x_data] #바이트를 문자열로 바꾸기
    print(x_data[0]) #태양광으로 충전되는 ‘아이폰X 테슬라’ (지디넷코리아=이정현 기자)도널드 트럼프 미국 대통령과 푸틴 러시아 대통령의 초상화를 새긴 황금 아이폰을 출시해 ...

    x_data = clean_korean_documents(x_data) #텍스트 정제
    transformer = TfidfVectorizer()
    transformer.fit(x_data)
    x_data = transformer.transform(x_data) #단어 카운트 가중치

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

    ##########모델 생성

    model = MultinomialNB(alpha=1.0)

    ##########모델 학습

    model.fit(x_train, y_train)

    ##########모델 검증

    model.score(x_test, y_test) #1.0

    if request.method=='POST':
        message = request.form['message']
        data = [message]
        x_test = np.array(data)
        x_test = clean_korean_documents(x_test) #텍스트 정제
        x_test = transformer.transform(x_test) #단어 카운트 가중치
        my_prediction = model.predict(x_test)
    return render_template('result.html',prediction=my_prediction)

if __name__=='__main__':
    app.run(host='127.0.0.1',debug=True)
