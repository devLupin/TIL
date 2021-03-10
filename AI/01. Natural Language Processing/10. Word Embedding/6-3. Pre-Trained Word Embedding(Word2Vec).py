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


import gensim

file_path = "C:\\Users\\devLupin\\Desktop\\GoogleNews-vectors-negative300.bin.gz"
word2vec_model = gensim.models.keyedvectors._load_word2vec_format(file_path, binary=True)

print(word2vec_model.vectors.shape)

embedding_matrix = np.zeros((vocab_size, 300))  # 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성
np.shape(embedding_matrix)


# 특정 단어를 입력하면 해당 단어의 임베딩 벡터를 리턴
def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, i in t.word_index.items() :
    temp = get_vector(word) # 단어(key) 해당되는 임베딩 벡터의 300개의 값(value)를 임시 변수에 저장
    if temp is not None:
        embedding_matrix[i] = temp # 해당 단어 위치의 행에 벡터의 값을 저장


print('단어 nice의 정수 인덱스 :', t.word_index['nice'])
print(word2vec_model['nice'])
print(embedding_matrix[1])


# model training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)