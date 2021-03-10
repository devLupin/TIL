"""
    케라스는 훈련 데이터의 단어들에 대해 워드 임베딩을 수행하는 Embedding()을 제공
    
    임베딩 층의 입력으로 사용하기 위해서 입력 시퀸스의 각 단어들은 모두 정수 인코딩이 되어있어야 한다.
        - 특정 단어와 맵핑되는 정수를 인덱스로 가지는 테이블로부터 임베딩 벡터 값을 가져오는 룩업 테이블
        - 룩업 테이블은 단어 집합의 크기만큼의 행을 가지므로 모든 단어는 고유한 임베딩 벡터를 지님.
"""

"""
    케라스 임베딩 층 구현 코드
    
    v = Embedding(20000, 128, input_length=500)
    # vocab_size = 20000
    # output_dim = 128
    # input_length = 500
    
    vocab_size : 텍스트 데이터 전체 단어 집합의 크기
    output_dim : 워드 임베딩 후의 임베딩 벡터의 차원
    input_length : 입력 시퀀스의 길이. 만약 갖고있는 각 샘플의 길이가 500개의 단어로 구성되어있다면 이 값은 500.
    
    Embedding(number of samples, input_length)
        - 2D 정수 텐서를 입력받음.
        - 각 sample은 정수 인코딩 된 결과로, 정수의 시퀸스
        - 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimentionality)인 3D 실수 텐서를 리턴
"""



""" Keras Embedding() """
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## 각각 문장과 레이블
sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]     # 1 is true

t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index) + 1

X_encoded = t.texts_to_sequences(sentences)     # 토큰화

# 정수 인코딩
max_len = max(len(l) for l in X_encoded)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)


# 이후부터 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
"""
    Flatten()
        - 하나를 제외한 모든 차원을 제거
        - 배치 차원을 포함하지 않고 텐서에 포함된 요소의 수와 동일한 모양을 갖도록 텐서 재구성
"""

model = Sequential()
model.add(Embedding(vocab_size, 4, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)