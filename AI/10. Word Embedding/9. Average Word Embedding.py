# -*- coding: euc-kr -*- 

"""
    특정 문장 내의 단어들이 임베딩 벡터들의 평균이 그 문장의 벡터가 될 수 있다.
    
    임베딩이 잘 된 상황에서는 단어 벡터들의 평균만으로 텍스트 분류를 수행할 수 있다.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import imdb

"""
    GlobalAveragePooling1D
        - 고정된 크기의 출력 벡터 리턴
        - 모든 벡터들의 평균 리턴
        - Embedding() 다음에 사용하면, 해당 문장의 모든 단어 벡터들의 평균 벡터 획득 가능
        - 입력으로 사용되는 리뷰에 포함된 단어 개수가 변경되더라도 같은 크기의 벡터로 처리 가능
        - ex) 입력으로 shape가 (25000, 256, 16)인 배열 사용
            두번째 차원(리뷰당 단어개수 256개) 방향으로 평균을 구하여 shape가 (25000, 16)인 배열을 생성
    EaryStopping
        - 무조건 Epoch을 많이 돌린 후 특정 시점에서 멈추는 것
    ModelCheckpoint
        - 모델을 저장할 때 사용되는 콜백함수
    imdb
        - 텐서플로우 영화 데이터셋
        - 정수 인코딩까지의 전처리가 진행되어 있어, 단어 집합을 만들고 정수 인코딩하는 과정 불필요
"""

vocab_size = 20000

# 등장 빈도수가 20,000등이 넘는 데이터만 불러옴.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print('훈련용 리뷰 개수 :',len(x_train))
print('테스트용 리뷰 개수 :',len(x_test))

print('훈련용 리뷰의 평규 길이: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('테스트용 리뷰의 평균 길이: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))


""" (훈련용, 테스트용) 리뷰의 평균 길이가 (238, 230) 이므로 400으로 패딩 """
max_len = 400
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
print('x_train의 크기(shape) :', x_train.shape)
print('x_test의 크기(shape) :', x_test.shape)



""" 임베딩 벡터를 평균으로 사용하는 모델 설계 """
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_len))
model.add(GlobalAveragePooling1D())     # 모든 단어 벡터의 평균
model.add(Dense(1, activation='sigmoid'))

"""
    patience
        - 성능이 증가하지 않는다고, 그 순간 멈추는 것은 효과적이지 않을 수 있음.
        - 성능이 증가하지 않는 epoch을 몇번이나 허용할 것인가 결정
"""
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('embedding_average_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# 훈련 데이터의 20%를 검증 데이터로 사용하고, 총 10회 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.2)

loaded_model = load_model('embedding_average_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))