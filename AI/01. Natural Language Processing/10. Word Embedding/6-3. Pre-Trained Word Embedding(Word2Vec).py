from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## ���� ����� ���̺�
sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]     # 1 is true

t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index) + 1

X_encoded = t.texts_to_sequences(sentences)     # ��ūȭ

# ���� ���ڵ�
max_len = max(len(l) for l in X_encoded)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)


import gensim

file_path = "C:\\Users\\devLupin\\Desktop\\GoogleNews-vectors-negative300.bin.gz"
word2vec_model = gensim.models.keyedvectors._load_word2vec_format(file_path, binary=True)

print(word2vec_model.vectors.shape)

embedding_matrix = np.zeros((vocab_size, 300))  # �ܾ� ���� ũ���� ��� 300���� ���� ������ ��� ����
np.shape(embedding_matrix)


# Ư�� �ܾ �Է��ϸ� �ش� �ܾ��� �Ӻ��� ���͸� ����
def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, i in t.word_index.items() :
    temp = get_vector(word) # �ܾ�(key) �ش�Ǵ� �Ӻ��� ������ 300���� ��(value)�� �ӽ� ������ ����
    if temp is not None:
        embedding_matrix[i] = temp # �ش� �ܾ� ��ġ�� �࿡ ������ ���� ����


print('�ܾ� nice�� ���� �ε��� :', t.word_index['nice'])
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