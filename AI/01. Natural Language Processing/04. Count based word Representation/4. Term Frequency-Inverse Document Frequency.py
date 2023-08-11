"""
    TF-IDF(단어 빈도-역 문서 빈도, Term Frequency Inverse)
        - 단어의 빈도와 역 문서 빈도를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 부여하는 방법
        - 우선 DTM을 만든 후, TF-IDF 가중치를 부여
        - 문서의 유사도, 검색 결과의 중요도, 특정 단어의 중요도를 구하는 작업 등에 사용
        
"""

import pandas as pd     # For dataframe
from math import log

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
N = len(docs)

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()


def tf(t, d) :
    return d.count(t)

def idf(t) :
    df = 0
    
    for doc in docs :
        df += t in doc
    
    return log(N/(df+1))

def tfidf(t, d) :
    return tf(t, d) * idf(t)


result = []


# Get TF
for i in range(N) :
    result.append([])
    d = docs[i]
    
    for j in range(len(vocab)) :
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
"""
tf_
0    0   0   0   1    0   1   1   0    0
1    0   0   0   1    1   0   1   0    0
2    0   1   1   0    2   0   0   0    0
3    1   0   0   0    0   0   0   1    1
"""

result = []

# Get IDF
for j in range(len(vocab)) :
    t = vocab[j]
    result.append(idf(t))
    
idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
"""
idf_
          IDF
과일이  0.693147
길고   0.693147
노란   0.693147
먹고   0.287682
바나나  0.287682
사과   0.693147
싶은   0.287682
저는   0.693147
좋아요  0.693147
"""

result = []

# Get TF-IDF
for i in range(N) :
    result.append([])
    d = docs[i]
    
    for j in range(len(vocab)) :
        t = vocab[j]
        
        result[-1].append(tfidf(t, d))
        
tfidf_ = pd.DataFrame(result, columns=vocab)
"""
tfidf_
        과일이        길고        노란        먹고       바나나        사과        싶은        저는       좋아요
0  0.000000  0.000000  0.000000  0.287682  0.000000  0.693147  0.287682  0.000000  0.000000
1  0.000000  0.000000  0.000000  0.287682  0.287682  0.000000  0.287682  0.000000  0.000000
2  0.000000  0.693147  0.693147  0.000000  0.575364  0.000000  0.000000  0.000000  0.000000
3  0.693147  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.693147  0.693147
"""


### DTM, TF-IDF using scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]

vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())   # 코퍼스로부터 각 단어의 빈도 수를 기록
print(vector.vocabulary_)   # 각 단어의 인덱스 부여 상태


"""
    TfidfVectorizer는 보편적인 TF-IDF에서 조금 조정된 식을 사용
        - IDF의 로그항의 분자에 1을 더함.
        - 로그항에 1을 더함.
        - TF-IDF에 L2 정규화를 통해 값을 조정
            L2 정규화는 벡터 p, q의 유클리디안 거리
"""
from sklearn.feature_extraction.text import TfidfVectorizer     # TF-IDF 자동 계산
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)