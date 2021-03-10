# -*- coding: euc-kr -*- 

"""
    Ư�� ���� ���� �ܾ���� �Ӻ��� ���͵��� ����� �� ������ ���Ͱ� �� �� �ִ�.
    
    �Ӻ����� �� �� ��Ȳ������ �ܾ� ���͵��� ��ո����� �ؽ�Ʈ �з��� ������ �� �ִ�.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import imdb

"""
    GlobalAveragePooling1D
        - ������ ũ���� ��� ���� ����
        - ��� ���͵��� ��� ����
        - Embedding() ������ ����ϸ�, �ش� ������ ��� �ܾ� ���͵��� ��� ���� ȹ�� ����
        - �Է����� ���Ǵ� ���信 ���Ե� �ܾ� ������ ����Ǵ��� ���� ũ���� ���ͷ� ó�� ����
        - ex) �Է����� shape�� (25000, 256, 16)�� �迭 ���
            �ι�° ����(����� �ܾ�� 256��) �������� ����� ���Ͽ� shape�� (25000, 16)�� �迭�� ����
    EaryStopping
        - ������ Epoch�� ���� ���� �� Ư�� �������� ���ߴ� ��
    ModelCheckpoint
        - ���� ������ �� ���Ǵ� �ݹ��Լ�
    imdb
        - �ټ��÷ο� ��ȭ �����ͼ�
        - ���� ���ڵ������� ��ó���� ����Ǿ� �־�, �ܾ� ������ ����� ���� ���ڵ��ϴ� ���� ���ʿ�
"""

vocab_size = 20000

# ���� �󵵼��� 20,000���� �Ѵ� �����͸� �ҷ���.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print('�Ʒÿ� ���� ���� :',len(x_train))
print('�׽�Ʈ�� ���� ���� :',len(x_test))

print('�Ʒÿ� ������ ��� ����: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('�׽�Ʈ�� ������ ��� ����: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))


""" (�Ʒÿ�, �׽�Ʈ��) ������ ��� ���̰� (238, 230) �̹Ƿ� 400���� �е� """
max_len = 400
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
print('x_train�� ũ��(shape) :', x_train.shape)
print('x_test�� ũ��(shape) :', x_test.shape)



""" �Ӻ��� ���͸� ������� ����ϴ� �� ���� """
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_len))
model.add(GlobalAveragePooling1D())     # ��� �ܾ� ������ ���
model.add(Dense(1, activation='sigmoid'))

"""
    patience
        - ������ �������� �ʴ´ٰ�, �� ���� ���ߴ� ���� ȿ�������� ���� �� ����.
        - ������ �������� �ʴ� epoch�� ����̳� ����� ���ΰ� ����
"""
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('embedding_average_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# �Ʒ� �������� 20%�� ���� �����ͷ� ����ϰ�, �� 10ȸ �н�
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.2)

loaded_model = load_model('embedding_average_model.h5')
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))