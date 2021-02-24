# -*- coding: euc-kr -*- 

"""
    캐글에서 제공하는 데이터를 전처리하고, 바닐라 RNN을 이용한 스팸 메일 분류기 구현
"""

#%%
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", \
                            filename="spam.csv")
data = pd.read_csv('spam.csv',encoding='latin1')
# %%
# 불필요 행 삭제
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
# 레이블 0, 1로 변경
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data[:5]
# %%
data.info()
# %%
data.isnull().values.any()
# %%
# 중복 샘플을 제거한 샘플 수
data['v2'].nunique(), data['v1'].nunique()
# %%
data.drop_duplicates(subset=['v2'], inplace=True) # v2 열에서 중복인 내용이 있다면 중복 제거
# %%
# 레이블 값의 분포 시각화
data['v1'].value_counts().plot(kind='bar');
# %%
# 데이터 분리
X_data = data['v2']
y_data = data['v1']
print('메일 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # X의 각 행에 토큰화 수행
sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장
# 부여된 각 정수는 각 단어의 빈도수가 높을수록 낮은 정수가 부여
# %%
word_to_index = tokenizer.word_index

threshold = 2
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

"""등장 빈도가 지나치게 낮은 단어가 많으므로 단어 집합의 크기를 제한 할 수 있음.
tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1)"""
# %%
vocab_size = len(word_to_index) + 1

# 훈련, 테스트 데이터 분리 8:2
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
# %%
X_data = sequences
print('메일의 최대 길이 : %d' % max(len(l) for l in X_data))
print('메일의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
max_len = max(len(l) for l in X_data)
data = pad_sequences(X_data, maxlen = max_len)
print("훈련 데이터의 크기(shape): ", data.shape)
# %%
X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 1034개의 데이터만 저장
y_test = np.array(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1034개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 4135개의 데이터만 저장
y_train = np.array(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4135개의 데이터만 저장
# %%
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32)) # (단어집합 크기, 임베딩 벡터의 차원) 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))   # 이진 분류이므로 1개의 뉴런, 시그모이드 함수 사용

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)
# %%
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
# %%
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# 이 데이터는 에포크 5를 넘어가기 시작하면 검증 데이터의 오차가 증가하는 경향이 있음.