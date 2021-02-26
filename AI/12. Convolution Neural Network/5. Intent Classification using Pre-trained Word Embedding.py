""" 사전 훈련된 워드 임베딩을 이용한 의도 분류 """
import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import urllib.request

# data download
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_text.npy", \
                            filename="train_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_text.npy", \
                            filename="test_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_label.npy", \
                            filename="train_label.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_label.npy", \
                            filename="test_label.npy")

"""
    npy 파일을 원활하게 로드하기 위해 설정
    (Numpy에서 pickle 파일 허용 여부를 의미하는 allow_pickle 디폴트값이 False이기 때문에 True로 바꿔주는 과정
"""
old = np.load
np.load = lambda *a,**k: old(*a,allow_pickle=True,**k)

# load & save to list
# 차례대로, 훈련용 문장, 훈련용 레이블, 테스트 문장, 테스트 레이블
intent_train = np.load(open('train_text.npy', 'rb')).tolist()
label_train = np.load(open('train_label.npy', 'rb')).tolist()
intent_test = np.load(open('test_text.npy', 'rb')).tolist()
label_test = np.load(open('test_label.npy', 'rb')).tolist()

""" 
    정수 인코딩 
        - 레이블, 의도 문장 정수 인코딩
            1. 의도 문장에 대해 토큰화 수행 후, 단어 집합을 만듦.
            2. 이후 정수 인코딩을 수행하여 텍스트 -> 정수 시퀸스로 변환
"""
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_train)

label_train = idx_encode.transform(label_train) # 주어진 고유한 정수로 변환
label_test = idx_encode.transform(label_test) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(intent_train)    # 토큰화
sequences = tokenizer.texts_to_sequences(intent_train)  # 정수 인코딩

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

max_len = max(len(l) for l in sequences)


""" padding """
# 레이블의 경우 다중 클래스 분류를 위해 원-핫 인코딩
intent_train = pad_sequences(sequences, maxlen = max_len)
label_train = to_categorical(np.asarray(label_train))


""" train 순서 랜덤화 """
indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
intent_train = intent_train[indices]
label_train = label_train[indices]

n_of_val = int(0.1 * intent_train.shape[0])     # 검증 데이터의 수

X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]
X_test = intent_test
y_test = label_test



""" Use pre-trained word embedding """
embedding_dict = dict()

f = open(os.path.join('glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32') # 100개의 값을 가지는 array로 변환
    embedding_dict[word] = word_vector_arr
f.close()

embedding_dim = len(embedding_dict['respectable'])
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# 임베딩 테이블에 저장
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        



""" Intent classification using 1D CNN """
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate

filter_sizes = [2,3,5]
num_filters = 512
drop = 0.5

model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(model_input)

conv_blocks = []

for sz in filter_sizes:
    conv = Conv1D(filters = num_filters,
                    kernel_size = sz,
                    padding = "valid",
                    activation = "relu",
                    strides = 1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(drop)(z)
model_output = Dense(len(label_idx), activation='softmax')(z)

model = Model(model_input, model_output)

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=10,
                    validation_data = (X_val, y_val))


""" 테스트 데이터 평가 """
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.argmax(axis=-1) # 예측된 정수 시퀀스로 변환

y_predicted = idx_encode.inverse_transform(y_predicted) # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환
y_test = idx_encode.inverse_transform(y_test) # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환

print('accuracy: ', sum(y_predicted == y_test) / len(y_test))
print("Precision, Recall and F1-Score:\n\n", classification_report(y_test, y_predicted))