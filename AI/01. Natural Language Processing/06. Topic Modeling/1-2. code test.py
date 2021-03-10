"""
    scikit-learn 에서는 'Twenty Newsgroups' 라는 20개의 다른 주제를 가진 뉴스그룹 데이터를 제공
"""

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
# len(documents)  # 샘플 수 출력
# print(dataset.target_names)     # 카테고리 종류 출력

news_df = pd.DataFrame({'document':documents})

news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")    # 특수 문자 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))   # 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())      # 전체 단어에 대한 소문자 변환

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])   # # 불용어 제거


"""
    불용어 제거를 위한 토큰화 작업 수행함.
    TfidVectorizer는 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용하므로, 역토큰화(Detokenization) 수행
"""
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # 상위 1,000개의 단어를 보존 
max_df = 0.5, 
smooth_idf=True)

# TF-IDF 행렬 생성
X = vectorizer.fit_transform(news_df['clean_doc'])      # 각 단어의 빈도수 기록


### TF-IDF 행렬을 사이킷런의 절단된 SVD를 사용하여 다수의 행렬로 분해 ###
from sklearn.decomposition import TruncatedSVD

"""
    뉴스그룹 데이터가 20개의 카테고리이므로 20개의 토픽을 가졌다고 가정
    토픽의 숫자는 n_componets 파라미터로 지정 가능
"""
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)      # (*토픽의 수 X 단어의 수) 크기를 가짐

terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어 저장

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
        # 20개 행의 각 1000개의 열 중 가장 값이 큰 5개의 값을 찾아서 단어로 출력
        
get_topics(svd_model.components_,terms)