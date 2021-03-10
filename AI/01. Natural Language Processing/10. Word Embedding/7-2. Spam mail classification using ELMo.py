"""
    ELMo�� �ټ��÷ο� 2.0���� ��� �Ұ���
    
    �ش� �ڵ�� �ټ��÷ο� 1���������� ��� �����ϸ�
    �ӽ� �������� Colab���� �����Ͽ���.
"""

%tensorflow_version 1.x     # �ټ��÷ο� ������ 1�������� ����

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# �ټ��÷ο� ���κ��� ELMo�� �ٿ�ε�

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# ���� ���� �з��ϱ� ������ �ٿ�ε�
urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')
data[:5]

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])   # v1���� �ִ� ham�� spam ���̺��� ���� ���� 0�� 1�� �ٲٰ� y_data�� ����
y_data = list(data['v1'])
X_data = list(data['v2'])

# �Ʒ�, �׽�Ʈ ������ 8:2�� ����
print(len(X_data))
n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)
print(n_of_train)
print(n_of_test)

# �Ʒ� �����Ϳ� �׽�Ʈ �������� ������ �Ͽ� �����͸� ����
X_train = np.asarray(X_data[:n_of_train])
y_train = np.asarray(y_data[:n_of_train])
X_test = np.asarray(X_data[n_of_train:])
y_test = np.asarray(y_data[n_of_train:])

""" ELMo�� �ټ��÷ο� ���κ��� ������ ���̱� ������ �ɶ󽺿��� ����ϱ� ���ؼ��� �ɶ󽺿��� ����� �� �ֵ��� ��ȯ���ִ� �۾����� �ʿ� """
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# �������� �̵��� �ɶ� �� �ټ��÷ο� �� �ɶ󽺰� �ǵ��� �ϴ� �Լ�

# �� ����
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
""" �߰��� �ټ��÷ο��� ���̾ ����ϰ��� �Ѵٸ� �ټ��÷ο� ���̾ �ɶ��� Lambda ���̾�� �����־�� ��. """
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1, batch_size=60)
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (model.evaluate(X_test, y_test)[1]))