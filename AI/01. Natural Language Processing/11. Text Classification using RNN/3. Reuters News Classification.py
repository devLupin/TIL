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
        (1) num_words : �� �� �� ����� ����� ����? ��ü ����� None
        (2) test_split : �׽�Ʈ ������ ����
"""

num_classes = max(y_train) + 1  # 0���� ī�װ� ���� �ο��ϱ� ����
# %%
print(X_train[0]) # ù��° �Ʒÿ� ���� ���
print(y_train[0]) # ù��° �Ʒÿ� ���� ����� ���̺�
# %%
print('���� ����� �ִ� ���� :{}'.format(max(len(l) for l in X_train)))
print('���� ����� ��� ���� :{}'.format(sum(map(len, X_train))/len(X_train)))

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
print("�� ���̺� ���� �󵵼�:")
print(np.asarray((unique_elements, counts_elements)))
label_cnt=dict(zip(unique_elements, counts_elements))
print(label_cnt)
# �Ʒ��� ��� ����� ���� �����Ͽ� ���ķ� ����ʹٸ� ���� label_cnt�� ���
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
# �󵵰� ���� 1000�� �ܾ� ���
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
model.add(Embedding(1000, 120))     # (�ܾ� ������ ũ��, �Ӻ��� ������ ����)
model.add(LSTM(120))    # 120�� �޸� ���� ���� ������ ũ��(hidden_size)
model.add(Dense(46, activation='softmax'))  # 46 categoris
# %%
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# monitor='val_acc : ���� ���� ��Ȯ������ ������ ��쿡�� ���� ����
# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# ���� Ŭ���� �з� �����̹Ƿ� ũ�ν� ��Ʈ���� �ս� �Լ� ���
# �̴� ���� �������� �������� ���ؼ� �� Ȯ�� ���� ������ �Ÿ��� �ּ�ȭ�ϵ��� �Ʒ�
# %%
history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))
# fit �� ��谡 ���� �Ʒ��� ���� ������, �������� ��Ȯ���� loss�� ����Ͽ� �������� �Ǵ��ϱ� ���� �뵵�� ���
# %%
loaded_model = load_model('best_model.h5')
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# %%
"""
�Ʒ� ������, ���� ������ �ս� �ð�ȭ
"""
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()