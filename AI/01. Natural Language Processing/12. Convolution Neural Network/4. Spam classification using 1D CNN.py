import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')

print(len(data))

data[:5]

# 불필요 열 제거
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data[:5]

data['v2'].nunique(), data['v1'].nunique()

data.drop_duplicates(subset=['v2'], inplace=True)

data['v1'].value_counts().plot(kind='bar');

print(data.groupby('v1').size().reset_index(name='count'))

X_data = data['v2']
y_data = data['v1']

vocab_size = 1000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_data)  # 토큰화
sequences = tokenizer.texts_to_sequences(X_data)    # 단어를 숫자, 인ㄷ게스로 변환

print(sequences[1])

# train : test = 8 : 2
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)

X_data = sequences
max_len = max(len(l) for l in X_data)
data = pad_sequences(X_data, maxlen=max_len)
print(data.shape)

# Data split to train, test
X_test = data[n_of_train:]
y_test = np.array(y_data[n_of_train:])
X_train = data[:n_of_train]
y_train = np.array(y_data[:n_of_train])


from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

history = model.fit(X_train, y_train, epochs = 10, batch_size=64, validation_split=0.2, callbacks=[es, mc])
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))