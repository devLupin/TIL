"""
    Supervised Learning(지도 학습)
        - 기계를 학습시키기 위해 훈련 데이터셋, 테스트 데이터셋으로 분류하여 학습
        - 훈련 데이터셋은 훈련을 위한 데이터이며, 이것이 기계의 Accuracy가 됨.
        - 테스트 데이터셋은 훈련된 기계를 평가하기 위한 데이터
        - 각 데이터 셋은 X(문제), Y(정답)을 지닌다.
"""



### X, Y 분류 ###


# zip 함수 이용
# zip 함수는 시퀸스 자료형에서 각 순서에 등장하는 원소들끼리 묶어주는 역할
x, y = zip(['a', 1], ['b', 2], ['c', 3])

sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 2D 텐서로 표현
x, y = zip(*sequences) # *를 추가


# 데이터프레임을 이용한 분류
import pandas as pd

values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
x=df['메일 본문']
y=df['스팸 메일 유무']


# Numpy 이용
import numpy as np
ar = np.arange(0, 16).reshape((4, 4))
"""
    : 의 의미는 전체라는 의미
    :n 의 의미는 0 ~ n-1번째 까지
    
    ar[x, y] 에서 x는 행, y는 열
"""



### 테스트 데이터 분류 ###


# 사이킷런 이용
from sklearn.model_selection import train_test_split        # 학습용 테스트와 테스트용 데이터를 분리하는 메서드
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=1234)
"""
    x : 독립 변수 데이터(배열, 데이터 프레임)
    y : 종속 변수 데이터, 레이블 데이터
    test_size : 테스트 데이터 개수 지정. 1보다 작은 실수라면 비율을 나타냄.
    train_size : 학습용 데이터의 개수를 지정. 1보다 작은 실수라면 비율을 나타냄.
    random_state : 난수 시드
"""

x, y = np.arange(10).reshape((5, 2)), range(5)  # 임시로 x, y 가 이미 분리된 형태를 사용함.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1234)
#3분의 1만 test 데이터로 지정.
#random_state 지정으로 인해 순서가 섞인 채로 훈련 데이터와 테스트 데이터가 나눠진다.