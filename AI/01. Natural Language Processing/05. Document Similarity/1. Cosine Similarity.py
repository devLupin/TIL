"""
    문서 유사도(Document Similarity)
        - 주로 문서들 간에 동일한 단어 또는 비슷한 단어가 얼마나 공통적으로 많이 사용되었는지에 의존
        
    코사인 유사도(Cosine Similarity)
        - BoW, DTM, TF-IDF, Word2Vec 같은 표현 방법에 대해서 코사인 유사도를 이용하여 문서의 유사도를 구하는 것이 가능
        - 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도
        - 두 벡터의 방향이 완전히 동일한 경우 1, 90도 각을 이루면 0, 180도로 반대 방향을 가지면 -1의 값. 즉, -1 이상 1 이하의 값을 가지며 1에 가까울수록 유사도가 높다
        - 문서의 길이가 다른 상황에서 비교적 공정한 비교
        - 코사인 유사도는 유사도를 구할 때, 벡터의 크기가 아닌 벡터의 방향(패턴)에 초점을 둔다.
"""


### Cosine similarity using Numpy ###

from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B) :
    return dot(A, B) / (norm(A) * norm(B))


# document BoW
doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

"""
cos_sim(doc1, doc2)
>> 0.6666666666666667

cos_sim(doc1, doc3)
0.6666666666666667

cos_sim(doc2, doc3)
1.0000000000000002
"""



### recommend system using similarity ###

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('c:\\Users\\devLupin\\Desktop\\TIL\\AI\\5. Document Similarity\movies_metadata.csv', low_memory=False)

# data.head(20000)

data['overview'].isnull().sum()     # overview 열에 Null 값이 있는지?
data['overview'] = data['overview'].fillna('')  # overview에서 Null 값을 가진 경우에는 Null 값을 제거

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(data['overview'])    # overview에 대해서 tf-idf 수행
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)      # Get cosine similarity

indices = pd.Series(data.index, index=data['title']).drop_duplicates()      # 중복 값 제거


# overview가 유사한 10개의 영화를 찾아내는 함수, 코사인 유사도 이용
def get_recommendations(title, cosine_sim=cosine_sim) :
    idx = indices[title]    # get index from title
    
    sim_scores = list(enumerate(cosine_sim[idx]))   # get similarlity about all movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]   ## Most similarity about 10
    
    movie_indices = [i[0] for i in sim_scores]
    
    return data['title'].iloc[movie_indices]    # 행을 추출하는데 숫자 계열이 아닌 가상 인덱스 위치를 사용하여 행 추출

print(get_recommendations('The Dark Knight Rises'))