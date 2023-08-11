# -*- coding: euc-kr -*- 

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

file_path = "C:\\Users\\devLupin\\Desktop\\"
file_name = "data.csv"

df = pd.read_csv(file_path + file_name)

""" desc 열의 전처리 수행 """
def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)    # ord는 문자의 아스키 코드 값을 돌려주는 함수

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# 구두점 제거
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# apply는 복잡한 연산을 vectorizing 할 수 있는 함수
df['cleaned'] = df['Desc'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)

""" 전처리 과정에서 빈 값이 생긴 행이 있다면, nan 값으로 변환 후 해당 행 제거 """
df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]  # 누락되지 않은 값을 찾아냄.

""" 토큰화 수행 """
corpus = []
for words in df['cleaned']:
    corpus.append(words.split())



""" 사전 훈련된 워드 임베딩 사용 """
urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
                            filename="GoogleNews-vectors-negative300.bin.gz")

word2vec_model = Word2Vec(size=300, window=5, min_count=2, workers=-1)
word2vec_model.build_vocab(corpus)
word2vec_model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin.gz', lockf=1.0, binary=True)
word2vec_model.train(corpus, total_examples = word2vec_model.corpus_count, epochs = 15)



""" 단어 벡터의 평균값 """
def vectors(document_list):
    document_embedding_list = []
    
    for line in document_list :
        doc2vec = None
        count = 0
        
        for word in line.split() :
            if word in word2vec_model.wv.vocab :
                count += 1
                
            # 문서의 모든 단어들의 벡터값을 더함.
            if doc2vec is None:
                doc2vec = word2vec_model[word]
            else:
                doc2vec = doc2vec + word2vec_model[word]
        
        # 모두 더한 단어 벡터의 값을 문서 길이로 나눔.
        if doc2vec is not None :
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
            
    return document_embedding_list

document_embedding_list = vectors(df['cleaned'])
print(len(document_embedding_list))



""" 추천 시스템 구현(유사한 5개의 책을 찾아냄) """
cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

def recommendations(title):
    books = df[['title', 'image_link']]
    # 책의 제목을 입력하면 해당 제목의 인덱스를 리턴받아 idx에 저장.
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()    
    idx = indices[title]

    # 입력된 책과 줄거리(document embedding)가 유사한 책 5개 선정.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    
    # 가장 유사한 책 5권의 인덱스
    book_indices = [i[0] for i in sim_scores]
    
    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    # iloc 은 위치 정수 기반 인덱싱 함수
    recommend = books.iloc[book_indices].reset_index(drop=True)
    
    fig = plt.figure(figsize=(20, 30))

    # 데이터프레임으로부터 순차적으로 이미지를 출력
    for index, row in recommend.iterrows():
        response = requests.get(row['image_link'])
        img = Image.open(BytesIO(response.content))
        fig.add_subplot(1, 5, index + 1)
        plt.imshow(img)
        plt.title(row['title'])
        
recommendations("The Da Vinci Code")
recommendations("The Murder of Roger Ackroyd")