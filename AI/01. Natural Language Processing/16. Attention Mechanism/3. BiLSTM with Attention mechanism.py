from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000  # 최대 단어 개수 제한
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))
print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train))/len(X_train)))

""" padding 작업, 정수 인코딩은 이미 진행됨. """
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query): # 단, key와 value는 같음.
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # self.V에 score를 적용하기 때문에 마지막 축에서 1을 얻음.
        # self.V를 적용하기 전 텐서의 shape == (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights



""" 모델 설계 """
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os

sequence_input = Input(shape=(max_len,), dtype='int32')     # input layer
embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero = True)(sequence_input)     # embedding layer

# bi-LSTM 사용
# 첫 번째 층
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(embedded_sequences)
# 두 번째 층
# 순방향 LSTM의 은닉 상태와 셀 상태를 forward_h, forward_c에 저장
# 역방향 LSTM의 은닉 상태와 셀 상태를 backward_h, backward_c에 저장
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
    (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
# lstm은 foward 방향과 backward 방향이 연결된 hidden state벡터가 모든 시점에 대해서 존재

# 양방향 LSTM을 사용할 경우에는 순방향 LSTM과 역방향 LSTM 각각 은닉 상태와 셀 상태를 가지므로
# 두 방향의 LSTM concatenate
state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

# 어텐션에서는 은닉 상태를 입력으로 사용
attention = BahdanauAttention(64) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout)    # 이진 분류이므로 1
model = Model(inputs=sequence_input, outputs=output)

# 시그모이드 함수이므로, binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 3, batch_size = 256, validation_data=(X_test, y_test), verbose=1)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))