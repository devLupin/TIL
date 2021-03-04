"""
    다중 선형 회귀
        - 입력되는 독립 변수가 2개이상인 것은, 입력 벡터의 차원이 2이상
"""

import numpy as np
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트

# 중간, 기말, 가산점
X=np.array([[70,85,11],[71,89,18],[50,80,20],[99,20,10],[50,10,10]])    # 3차원, input_dim=3
# 최종 성적
y=np.array([73,82,72,57,34])    # 1차원, output_dim=1

model=Sequential()
# Dense 레이어는 입력과 출력을 모두 연결해주며 입력과 출력을 연결해주는 가중치를 포함한다.
model.add(Dense(1, input_dim=3, activation='linear'))
sgd=optimizers.SGD(lr=0.00001)

model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])    # sgd(경사하강법), mse(평균제곱오차)

model.fit(X,y, batch_size=1, epochs=2000, shuffle=False)    # 2000번 오차 최소화 작업 시도

# 예측 작업, 기존 미사용 데이터
X_test=np.array([[20,99,10],[40,50,20]]) # 각각 58점과 56점을 예측해야함.
print(model.predict(X_test))
"""
>>
    [[58.08134 ]
     [55.734634]]
"""



"""
    다중 로지스틱 회귀
        - y를 결정하는 데 독립 변수 x가 2개 이상인 로지스틱 회귀
"""

### OR gate using Logistic Regression ###

import numpy as np

X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])    # input_dim=2
y=np.array([0, 1, 1, 1]) # output_dim=1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model=Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(optimizer='sgd' ,loss='binary_crossentropy',metrics=['binary_accuracy'])

model.fit(X,y, batch_size=1, epochs=800, shuffle=False)     # 800회 수행

print(model.predict(X))
"""
>>
    [[0.45521256]   # [0, 0]
     [0.84107596]   # [0, 1]
     [0.8577089 ]   # [1, 0]
     [0.97447586]]  # [1, 1]
"""