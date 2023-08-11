"""
    다층 퍼셉트론(MultiLayer Perceptron, MLP)
        - 은닉층이 1개 이상 추가된 신경망
        - 피드 포워드 신경망의 가장 기본적인 형태
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

t = Tokenizer()
t.fit_on_texts(texts)   # 정수 인덱스 부여

print(t.texts_to_matrix(texts, mode='count'))   # 문서단어행렬(Document-Term Matrix, DTM) 생성
"""
0번째 인덱스의 모든 열은 0이다.
>>
    [[0. 0. 1. 1. 1. 0. 0. 0. 0. 0.]
     [0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
     [0. 2. 0. 0. 0. 1. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]]
"""

print(t.texts_to_matrix(texts, mode = 'binary'))
"""
바이너리 모드는 단어가 있고 없고만 판단한다. 몇번 등장했는지는 관심없음.
>>
    [[0. 0. 1. 1. 1. 0. 0. 0. 0. 0.]
     [0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 1. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]]
"""

print(t.texts_to_matrix(texts, mode = 'tfidf').round(2)) # 둘째 자리까지 반올림하여 출력
"""
각 단어의 빈도에 자연 로그를 씌우고 1을 더한 값
>>
    [[0.   0.   0.85 0.85 1.1  0.   0.   0.   0.   0.  ]
     [0.   0.85 0.85 0.85 0.   0.   0.   0.   0.   0.  ]
     [0.   1.43 0.   0.   0.   1.1  1.1  0.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   1.1  1.1  1.1 ]]
"""

print(t.texts_to_matrix(texts, mode = 'freq').round(2)) # 둘째 자리까지 반올림하여 출력
"""
각 단어의 등장 횟수 / 각 문서에서 등장한 모든 단어의 개수의 총 합
>>
    [[0.   0.   0.33 0.33 0.33 0.   0.   0.   0.   0.  ]
     [0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.  ]
     [0.   0.5  0.   0.   0.   0.25 0.25 0.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33]]
"""