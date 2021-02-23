"""
    ELMo는 텐서플로우 2.0에서 사용 불가능
    
    해당 코드는 텐서플로우 1버전에서만 사용 가능하며
    임시 방편으로 Colab에서 실행하였음.
"""

%tensorflow_version 1.x     # 텐서플로우 버전을 1버전으로 설정

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# 텐서플로우 허브로부터 ELMo를 다운로드

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# 스팸 메일 분류하기 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')
data[:5]

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])   # v1열에 있는 ham과 spam 레이블을 각각 숫자 0과 1로 바꾸고 y_data에 저장
y_data = list(data['v1'])
X_data = list(data['v2'])

# 훈련, 테스트 데이터 8:2로 분할
print(len(X_data))
n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)
print(n_of_train)
print(n_of_test)

# 훈련 데이터와 테스트 데이터의 양으로 하여 데이터를 분할
X_train = np.asarray(X_data[:n_of_train])
y_train = np.asarray(y_data[:n_of_train])
X_test = np.asarray(X_data[n_of_train:])
y_test = np.asarray(y_data[n_of_train:])

""" ELMo는 텐서플로우 허브로부터 가져온 것이기 때문에 케라스에서 사용하기 위해서는 케라스에서 사용할 수 있도록 변환해주는 작업들이 필요 """
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# 데이터의 이동이 케라스 → 텐서플로우 → 케라스가 되도록 하는 함수

# 모델 설계
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
""" 중간에 텐서플로우의 레이어를 사용하고자 한다면 텐서플로우 레이어를 케라스의 Lambda 레이어로 감싸주어야 함. """
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1, batch_size=60)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))