#%%
from numpy.core.defchararray import encode
from tensorflow.keras.datasets import reuters
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
"""
    args
        (1) num_words : 빈도 수 몇 등까지 사용할 건지? 전체 사용은 None
        (2) test_split : 테스트 데이터 비율
"""

num_classes = max(y_train) + 1  # 0부터 카테고리 라벨을 부여하기 때문
# %%
print(X_train[0]) # 첫번째 훈련용 뉴스 기사
print(y_train[0]) # 첫번째 훈련용 뉴스 기사의 레이블
# %%
print('뉴스 기사의 최대 길이 :{}'.format(max(len(l) for l in X_train)))
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(12,5)
sns.countplot(y_train)
# %%
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))
label_cnt=dict(zip(unique_elements, counts_elements))
print(label_cnt)
# 아래의 출력 결과가 보기 불편하여 병렬로 보고싶다면 위의 label_cnt를 출력
# %%
word_to_index = reuters.get_word_index()
print(word_to_index)
# %%
index_to_word ={}
for key, value in word_to_index.items() :
    index_to_word[value] = key
# %%
print(index_to_word[0])
# %%
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in X_train[0]]))
# %%
"""
    Reuters News Classification using LSTM
"""
# %%
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
# %%
# 빈도가 높은 1000개 단어 사용
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
# %%
"""padding"""
max_len = 100
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
# %%
"""one-hot encoding"""
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# %%
model = Sequential()
model.add(Embedding(1000, 120))     # (단어 집합의 크기, 임베딩 벡터의 차원)
model.add(LSTM(120))    # 120은 메모리 셀의 은닉 상태의 크기(hidden_size)
model.add(Dense(46, activation='softmax'))  # 46 categoris
# %%
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# monitor='val_acc : 이전 모델의 정확도보다 좋아질 경우에만 모델을 저장
# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중 클래스 분류 문제이므로 크로스 엔트로피 손실 함수 사용
# 이는 모델의 예측값과 실제값에 대해서 두 확률 분포 사이의 거리를 최소화하도록 훈련
# %%
history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))
# fit 은 기계가 실제 훈련은 하지 ㅇ낳고, 에폭마다 정확도와 loss를 출력하여 과적합을 판단하기 위한 용도로 사용
# %%
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# %%
"""
훈련 데이터, 검증 데이터 손실 시각화
"""
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()