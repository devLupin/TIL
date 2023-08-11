# **SeqFace**

<hr>

[1] : "SeqFace: Make full use of sequence information for face recognition", in *CVPR 2018*



## Abstract

<hr>

- label smoothing regularization(LSR)
- discriminative sequence agent (DSA) loss
  - sequence data 이용



## Introduction

- 얼굴 특징을 학습하는 두가지 메서드

  - classification loss function
    - softmax
  - metric learning loss function
    - contrastive loss, triplet loss

- **identity data** : 같은 형태를 공유하는 학습 데이터셋

  - face image, identity 주석 필요
  - 많은 얼굴 시퀸스로 생성된 데이터의 각 시퀸스는 하나의 ID를 가진 여러 얼굴 포함
  - 하나의 identity에만 속할 수 있음.
  - intra-identity variation을 감소시키는 역할

- face feature를 identity, sequence data를 이용하여 판별력 학습

- sequence data는 학습 절차에서 활용됨.

  ![Fig 1](Fig/1.PNG?raw=true)

- chief classification loss(softmax loss)

  - inter-identity variations 최대화
  - intra-identity variations 최소화

- auxiliary loss(center loss)

  - intra-identity 축소
  - 상기 언급된 DSA loss 제안
    - 클래스 간 분산에 기여



## Proposed Approach

### 1. SeqFace Framework

- identity dataset, sequence dataset을 이용한 학습

- chief classification loss + auxiliary loss 공동으로 지도학습

- chief loss는 inter-identity feature의 차이를 크게 하고, intra-identity feature의 변동을 감소

- auxiliary loss는 intra-identity(intra-sequence) 변동을 감소

- loss formulated

  ![Eq 1](Eq/1.PNG?raw=true)

- 시퀀스 데이터의 input face는 classification loss의 클래스(identity)에 속할 수 없음.

  - identity 데이터만 처리할 수 있으며, 시퀸스 데이터는 무시됨.

- **loss에서 feature 압축을 진행하고 이에 따른 패널티를 부여하지 않으면, CNN에서 지도학습하여 시퀸스 데이터에서 구별되는 얼굴 특징을 학습할 수 있고, identity 데이터도 처리됨.**

  - **시퀸스, identity 내 압축에 영향**을 미치므로, auxiliary loss여야 함.

  - center loss function

    - Xk : k번째 학습 샘플
    - Cyk : y번째 특징의 클래스 중심

    ![Eq 2](Eq/2.PNG?raw=true)

- softmax loss는 시퀸스 데이터 무시

### 2. Label Smoothing Regularization(LSR)

- softmax + cross-entropy

  - 실제 값이 아닌 입력에 대한 처리
  - P(i) : 예측된 확률
  - q(i) : 실제 분포

  ![Eq 3](Eq/3.PNG?raw=true)

  ​	- q(i) : 0 또는 1로 입력 데이터가 어떤 클래스인지 명백하게 라벨링

  ![Eq 4](Eq/4.PNG?raw=true)

- L2-constraint 추가

  - regular SphereFace를 통해
  - 입력 feature는 처음 스칼라 파라미터 δ 에 의해 일반화
  - L2-SphereFace의 결정 경계
    - δ(cos mθ1 - cos θ2) = 0 for class 1 
    - δ(cos θ1 - cos mθ2) = 0 for class 2 

### 3. Discriminative Sequence Agent(DSA) loss

- k번째 학습 샘플의 특징과 n번째 클래스(identity) 중심 간 거리 계산

  ![Eq 6](Eq/6.PNG?raw=true)

- Xk, Cn이 일반화 된 공식

  ![Eq 7](Eq/7.PNG?raw=true)

- Xk, Cyk 간 거리 감소

- 클래스, 해당되지 않는 특징 간 거리 증가

- Discriminative loss

  - α(1, +INF) and β(1, +INF)  are two parameters to adjust the discriminative power  

  ![Eq 8](Eq/8.PNG?raw=true)

- Final loss

  - λ는 intra-class 축소, inter-class 분산의 균형을 위해 적용
  - p는 final loss를 계산하는데 n번째 클래스 중심이 사용될 확률
  - b(1, p)는 확률 p가 베르누이 분포를 따르는 것을 의미
    - 베르누이 분포 : 오직 두 가지의 결과만 일어난다고 할 때 그 값이 0과 1로 결정되는 확률 변수 X에 대해 P(X=0) = q, P(X=1) = p, 0<=p<=1, q=1-p를 만족하는 확률변수 X가 따르는 확률 분포

- **특징 Xk는 해당 identity의 feature center Cyk로 쪽으로 당겨지고, 각 iteration에서 무작위로 선택된 다른 identity의 feature center에서 멀어짐.**

- **k번째 샘플이 identity 데이터셋에 있는 경우 Xk는 다른 identity 및 모든 시퀸스의 feature center에서 멀리 밀어야 함.**

  - **그렇지 않으면 Xk는 identity의 feature center에서만 멀어짐.**



## Result

- Visualization of 2-D feature distribution for the MNIST test set 

  ![Fig 3](Fig/3.PNG?raw=true)

- Different parameter values on LFW

  ![Fig 3](Fig/3.PNG?raw=true)

- Face verification ACC on LFW

  - CASIA-WebFace dataset are randomly divided into two parts
    - dataset A (5,000 identities), dataset B(5,575 identities)

  ![Table 1](Table/1.PNG?raw=true)

- Verification ACCs(%) of different methods on LFW and YTF 

  ![Table 2](Table/2.PNG?raw=true)