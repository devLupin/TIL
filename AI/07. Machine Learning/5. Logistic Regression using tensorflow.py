import numpy as np
%matplotlib inline  # notebook에서 실행한 브라우저에서 바로 그림을 볼 수 있게 함.
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X=np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) #숫자 10부터 1

model=Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))      # 로지스틱 회귀 이므로 sigmoid

# 옵티마이저는 경사하강법 sgd 사용
sgd=optimizers.SGD(lr=0.01)
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피) 사용
# 참, 거짓 이므로 이진 분류 사용
model.compile(optimizer=sgd ,loss='binary_crossentropy',metrics=['binary_accuracy'])

# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도
model.fit(X,y, batch_size=1, epochs=200, shuffle=False)

"""
>>
    Epoch 1/200
    13/13 [==============================] - 1s 65ms/step - loss: 0.3375 - binary_accuracy: 0.8462
    ... 중략 ...
    Epoch 192/200
    13/13 [==============================] - 0s 1ms/step - loss: 0.0898 - binary_accuracy: 1.0000   # 정확도 100퍼센트 달성
    ... 중략 ...
    Epoch 200/200
    13/13 [==============================] - 0s 1ms/step - loss: 0.0883 - binary_accuracy: 1.0000
"""


print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))
"""
>>
    [[0.21071826]
     [0.26909265]
     [0.33673897]
     [0.41180944]
     [0.45120454]]
    [[0.86910886]
     [0.99398106]
     [0.99975663]
     [0.9999902 ]
     [1.        ]]
     
     # X 값이 5보다 작을 때는 0.5보다 작은 값을, X값이 10보다 클 때는 0.5보다 큰 값을 출력
"""