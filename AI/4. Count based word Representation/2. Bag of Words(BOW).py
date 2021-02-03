"""
    Bag of Words(BOW)
        - 단어의 등장 순서를 고려하지 않는 빈도수 기반 단어 표현 방법
        - 제작 과정
            (1) 각 단어에 고유한 정수 인덱스 부여
            (2) 각 인덱스 위치에 단어 토큰의 등장 횟수를 기록한 벡터 제작
        
        - 어떤 단어가 얼마나 등장했는지를 기준으로 문서가 어떤 성격의 문서인지를 판단하는 작업에 사용
        - 즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 적합
"""


# 입력된 문서에 단어 집합을 만들어 인덱스를 할당하고 BoW 제작
from konlpy.tag import Okt
import re

okt = Okt()
token = re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  # 정규 표현식을 이용한 '.' 제거
token = okt.morphs(token)   # 토큰화 작업 수행

word2index = {}
bow = []

for voca in token :
    if voca not in word2index.keys() :
        word2index[voca] = len(word2index)
        bow.insert(len(word2index) - 1, 1)      # default value is 1 becuase number of word less more 1.
    else :
        index = word2index.get(voca)    # 재등장 단어 인덱스 확보
        bow[index] = bow[index] + 1     # 단어의 개수를 세는 작업
        


### using CountVectorizer class in scikit-learn ###
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()  # 단어의 빈도를 카운트하여 벡터로 만듬
# CounterVectorizer는 길이가 2 이상인 문자에 대해서만 토큰으로 인식하므로, I는 삭제됨.
# 영어의 경우 띄어쓰기만으로 토큰화가 가능하지만, 한국어의 경우 조사 등의 이유로 BoW가 제대로 만들어지지 않음.

vector.fit_transform(corpus).toarray()  # 단어의 빈도 수 기록
vector.vocabulary_  # 각 단어의 인덱스 부여 현황


### Generate BoW before remove stop-word ###

# Method1. User defined
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
vect.fit_transform(text).toarray()
vect.vocabulary_

# Method2. Self stop-word provided by CountVectorizer
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
vect.fit_transform(text).toarray()
vect.vocabulary_

# Method3. stop-word Provided by NLTK
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
vect.fit_transform(text).toarray()
vect.vocabulary_