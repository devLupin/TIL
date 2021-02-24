# -*- coding: euc-kr -*- 

"""
    ĳ�ۿ��� �����ϴ� �����͸� ��ó���ϰ�, �ٴҶ� RNN�� �̿��� ���� ���� �з��� ����
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
# ���ʿ� �� ����
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
# ���̺� 0, 1�� ����
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data[:5]
# %%
data.info()
# %%
data.isnull().values.any()
# %%
# �ߺ� ������ ������ ���� ��
data['v2'].nunique(), data['v1'].nunique()
# %%
data.drop_duplicates(subset=['v2'], inplace=True) # v2 ������ �ߺ��� ������ �ִٸ� �ߺ� ����
# %%
# ���̺� ���� ���� �ð�ȭ
data['v1'].value_counts().plot(kind='bar');
# %%
# ������ �и�
X_data = data['v2']
y_data = data['v1']
print('���� ������ ����: {}'.format(len(X_data)))
print('���̺��� ����: {}'.format(len(y_data)))
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # X�� �� �࿡ ��ūȭ ����
sequences = tokenizer.texts_to_sequences(X_data) # �ܾ ���ڰ�, �ε����� ��ȯ�Ͽ� ����
# �ο��� �� ������ �� �ܾ��� �󵵼��� �������� ���� ������ �ο�
# %%
word_to_index = tokenizer.word_index

threshold = 2
total_cnt = len(word_to_index) # �ܾ��� ��
rare_cnt = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ������ ī��Ʈ
total_freq = 0 # �Ʒ� �������� ��ü �ܾ� �󵵼� �� ��
rare_freq = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ���� �󵵼��� �� ��

# �ܾ�� �󵵼��� ��(pair)�� key�� value�� �޴´�.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # �ܾ��� ���� �󵵼��� threshold���� ������
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('���� �󵵰� %s�� ������ ��� �ܾ��� ��: %s'%(threshold - 1, rare_cnt))
print("�ܾ� ����(vocabulary)���� ��� �ܾ��� ����:", (rare_cnt / total_cnt)*100)
print("��ü ���� �󵵿��� ��� �ܾ� ���� �� ����:", (rare_freq / total_freq)*100)

"""���� �󵵰� ����ġ�� ���� �ܾ �����Ƿ� �ܾ� ������ ũ�⸦ ���� �� �� ����.
tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1)"""
# %%
vocab_size = len(word_to_index) + 1

# �Ʒ�, �׽�Ʈ ������ �и� 8:2
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
# %%
X_data = sequences
print('������ �ִ� ���� : %d' % max(len(l) for l in X_data))
print('������ ��� ���� : %f' % (sum(map(len, X_data))/len(X_data)))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
max_len = max(len(l) for l in X_data)
data = pad_sequences(X_data, maxlen = max_len)
print("�Ʒ� �������� ũ��(shape): ", data.shape)
# %%
X_test = data[n_of_train:] #X_data ������ �߿��� ���� 1034���� �����͸� ����
y_test = np.array(y_data[n_of_train:]) #y_data ������ �߿��� ���� 1034���� �����͸� ����
X_train = data[:n_of_train] #X_data ������ �߿��� ���� 4135���� �����͸� ����
y_train = np.array(y_data[:n_of_train]) #y_data ������ �߿��� ���� 4135���� �����͸� ����
# %%
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32)) # (�ܾ����� ũ��, �Ӻ��� ������ ����) �Ӻ��� ������ ������ 32
model.add(SimpleRNN(32)) # RNN ���� hidden_size�� 32
model.add(Dense(1, activation='sigmoid'))   # ���� �з��̹Ƿ� 1���� ����, �ñ׸��̵� �Լ� ���

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)
# %%
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (model.evaluate(X_test, y_test)[1]))
# %%
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# �� �����ʹ� ����ũ 5�� �Ѿ�� �����ϸ� ���� �������� ������ �����ϴ� ������ ����.