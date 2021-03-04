### Euclidean distance ###

import numpy as np

def dist(x, y) :
    return np.sqrt(np.sum((x-y)**2))

# Define DTM
doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

"""
print(dist(doc1,docQ))
print(dist(doc2,docQ))
print(dist(doc3,docQ))

2.23606797749979    // 유클리드 거리의 값이 가장 작다는 것은 문서 간의 거리가 가장 가깝다는 것을 의미
3.1622776601683795
2.449489742783178
"""



### Jaccard similarity ###
"""
    - 합집합에서 교집합의 비율을 구한다면 두 집합 A, B의 유사도를 구할 수 있다.
    - 자카드 유사도는 0과 1 사이의 값을 가지며, 두 집합이 동일하다면 1, 공통 원소가 없다면 0을 갖는다.
    - (docA n docB) / (docA U docB)     문서 A,B 교집합 / 합집합
"""

# 두 문서 모두에서 등장한 단어는 apple과 banana 2개.
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화 수행
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

union = set(tokenized_doc1).union(set(tokenized_doc2))  # 합집합
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))    # 교집합

print(len(intersection)/len(union))     # Get Jaccard similarity