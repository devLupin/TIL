# **CosFace[1]**

<hr>

[1] : "CosFace: Large Margin Cosine Loss for Deep Face Recognition" in *CVPR 2018*



## Abstract

<hr>

- 전통적인 deep CNNs의 softmax loss는 판별 능력이 때때로 부족
- 방사형(radial) 변화 제거를 위해 feature와 weight vectorL2 normalizing하여 cosine loss를 reform한 손실 함수 제안
- normalization과 cosine decision margin 최대화를 통해 클래스 내 변화 최소화, 클래스 간 변화 최대화 달성



## Introduction

<hr>

- idea : 클래스 내 다양성 최대, 클래스 간 다양성 최소

- angular space에서 decision margin을 최대화하는 cosine margin term m 기반

- radial variation 제거를 위해 feature와 weight vector를 normalizing 함으로써 cosine loss reform

- 클래스 간 cosine margin을 최대화하여 판별 특징을 학습하기 위해 normalized 된 특징을 입력으로 취함.

- Decision boundary는 cos(θ1) - m = cos(θ2)로부터 주어짐.

  - θ는 feature, weight 사이의 각도

- cosine space에서 decision margin 정의

- phase

  - training : large cosine margin으로 feature 학습
  - testing : ConvNet으로부터 face feature 추출

  ![Fig 1](Fig/1.PNG?raw=true)



## Proposed Approach

<hr>
### 1. Large Margin Cosine Loss(LMCL)


- Softmax

  - ground-truth class의 사후 확률을 최대화함으로써, 다른 클래스로부터 feature 분리

    ![Eq 1](Eq/1.PNG?raw=true)

  - 간소화를 위해 bias 값을 고정

    ![Eq 2](Eq/2.PNG?raw=true)

- L2 normalization함으로써, || Wj || = 1로 고정

  - testing stage : testing face pair의 점수는 두 feature vector 간 cosine similarity에 따라 계산되기도 함.
  - training stage : x는 scoring function에 기여하지 않아, || x || = s로 고정
    - radial direction의 변화 제거

- 사후 확률은 각도의 cosine에만 의존

  ![Eq 3](Eq/3.PNG?raw=true)

- 논문에서 제안한 Loss는 cos(θ1) - m > cos(θ2) and cos(θ2) - m > cos(θ1) 이 요구됨.

  - m > 0은 cosine margin의 크기를 제어하기 위해 고정

  ![Eq 4](Eq/4.PNG?raw=true)


### 2. Normalization on Features

- normalization scheme

  - cosine loss 공식을 도출
  - radial direction에서 변화 제거

- weight vector, feature vector 동시에 normalization

- Respects

  - feature normalization 없는 original softmax loss는 feature vector의 L2-norm 과 각도의 cosine 값을 암묵적으로 학습
  - LMCL은 cosine value에만 의존하는 학습과 같이 같은 L2-norm을 갖기 위해 전체 feature vector set 요구
  - 같은 클래스의 feature vector는 클러스터
    - 다른 클래스의 것들은 하이퍼스피어의 표면에서 분리

- cos(θi) - m이 cos(θi)보다 작으면 손실을 최소화하기 위한 || x ||를 줄여야 하므로 최적화가 저하됨.

  - adaptively learning 보다 scaling parameter s를 수정하는 것이 더 유리함.
    - s는 적절하게 큰 값으로 설정
    - 더 낮은 training loss로 더 나은 feature 생성

- 예상되는 큰 margin으로 feature를 학습하기 위한 충분한 hypersphere를 보장하기 위해 충분히 큰 s 필요

  - Pw : 예상되는 클래스 중심의 최소 사후확률

  - C : 전체 클래스의 수

    ![Eq 6](Eq/6.PNG?raw=true)

  - 최적의 Pw를 위해서는 s가 지속적으로 확대되어야 함.

  - Pw를 고정하여 더 많은 클래스를 처리하려면 s가 더 커야 함.

  - 클래스 내 거리가 작고 클래스 간 거리가 큰 특징을 임베딩하려면 반지름 s가 큰 하이퍼스피어 필요

### 3. Theoretical Analysis

- cos θ1 - cos θ2 = m 에 의해 클래스 간 분산은 확대되고 클래스 내 분산은 축소

- maximum angular margin은 W1과 W2 사이의 각도에 따름

- 모든 feature vector는 클래스 i의 weight vector와 동일

- 모든 feature vector는 클래스 중심에 존재하게 됨.

  ![Fig 3](Fig/3.PNG?raw=true)

  