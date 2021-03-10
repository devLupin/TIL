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

file_path = "C:\\Users\\devLupin\\Desktop\\glove.6B.100d.txt"


import numpy as np

embedding_dict = dict()
f = open(file_path, encoding='utf8')

for line in f :
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32')  # 100개의 값을 가지는 array로 변환
    embedding_dict[word] = word_vector_arr
    
f.close()

# print(embedding_dict['hello'])

embedding_matrix = np.zeros((vocab_size, 100))      # vocab_size * 100 크기의 행렬
np.shape(embedding_matrix)

print(t.word_index.items())

for word, i in t.word_index.items() :
    temp = embedding_dict.get(word)
    
    if temp is not None :
        embedding_matrix[i] = temp      # 임시 변수의 값을 단어와 맵핑되는 인덱스 행에 삽입
        
        

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
# 사전 훈련된 워드 임베딩을 100차원의 값인 것으로 사용하고 있기 때문에 임베딩 층의 output_dim의 인자 값으로 100
# trainable=False 별도로 더 이상 훈련을 하지 않는다.
e = Embedding(vocab_size, output_dim=100, weights=[embedding_matrix], input_length=max_len, trainable=False)

model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)