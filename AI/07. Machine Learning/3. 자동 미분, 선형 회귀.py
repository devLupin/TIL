### 자동 미분 ###

import tensorflow as tf

w = tf.Variable(2.)

def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:     # 자동 미분 기능 수행
    z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)



### 자동미분을 이용한 선형 회귀 ###

# 가중치, 편향 선언
W = tf.Variable(4.0)
b = tf.Variable(1.0)

#텐서플로우 즉시 실행
@tf.function
def hypothesis(x) :
    return W*x + b

x_test = [3.5, 5, 5.5, 6]   # 임의의 값 입력
print(hypothesis(x_test).numpy())

@tf.function
def mse_loss(y_pred, y):
    # 두 개의 차이값을 제곱을 해서 평균을 취한다.
    # tf.square(x) - 제곱 함수
    # tf.reduce_mean(x) - 평균
    return tf.reduce_mean(tf.square(y_pred - y))


X=[1,2,3,4,5,6,7,8,9] # 공부하는 시간
y=[11,22,33,44,53,66,77,87,95] # 각 공부하는 시간에 맵핑되는 성적

optimizer = tf.optimizers.SGD(0.01)     # 경사하강법 사용, learning rate는 0.01

for i in range(301):
    with tf.GradientTape() as tape:
        # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
        y_pred = hypothesis(X)
        
        # 평균 제곱 오차를 계산
        cost = mse_loss(y_pred, y)
        
        # 손실 함수에 대한 파라미터의 미분값 계산
        gradients = tape.gradient(cost, [W, b])
        
        # 파라미터 업데이트
        # zip 함수는 동일한 개수의 자료형 끼리 묶어주는 내장함수
        optimizer.apply_gradients(zip(gradients, [W, b]))
        
    if i % 10 == 0:
        print("epoch : {:3} | W의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, W.numpy(), b.numpy(), cost))
        
        
        

"""
    Keras로 구현하는 선형 회귀
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X=[1,2,3,4,5,6,7,8,9] # 공부하는 시간
y=[11,22,33,44,53,66,77,87,95] # 각 공부하는 시간에 맵핑되는 성적

model = Sequential()    # 모델 만들기
model.add(Dense(1, input_dim=1, activation='linear'))       # 입력 x의 차원(input_dim)은 1, 출력 y의 차원도 1. 선형 회귀이므로 activation은 'linear'

sgd = optimizers.SGD(lr=0.01)   # 경사하강법. 학습률은 0.01

model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])    # 손실 함수는 평균제곱오차(mse) 사용

model.fit(X,y, batch_size=1, epochs=300, shuffle=False)     # 주어진 x, y 데이터에 대해서 오차 최소화 작업 300번 시도


# 그래프 그리기
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X,y, 'k.')


# 특정 입력을 넣을 경우 예측되는 점수 출력
print(model.predict([9.5]))