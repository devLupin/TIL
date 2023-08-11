# **Qamface**

[1] : "Qamface: Quadratic Additive Angular Margin Loss For Face Recognition", International Conference on Image Processing (ICIP) 2020



## Abstract

- ArcFace의 한계점 지적
  - Non-monotonic logit, gradient curve, 부적절한 loss value
- QAMFace
  - cosine function 대신 quadratic function을 통해 각도 값을 취함.
  - 무시가능한 계산 오버헤드가 추가됨.

## Introduction

- 본 논문은 softmax 기반 손실함수를 다룸.
  - triplet loss의 얼굴 triplet 조합 폭발로 인해 semi-hard mining으로 훈련이 어려움.
- margin based softmax ArcFace
  - 적응형 마진 패널티를 추가하여 SOTA 성능 획득
  - 그러나 두가지 한계점 존재
    - gradient curve가 비단조적이고 decision margin의 overlap 부분 존재
    - 같은 클래스 간 마진에서 손실은 증가하는데 target angle은 감소함.

## Related works

- ArcFace

  - adds an additive angular margin penalty m between xi and Wyi

    ![Eq 3](Eq/3.PNG?raw=true)

## Proposed method

### 1. QAMFace

 - first constructed a quadratic function: f(x) = (2π - x)^2

 - additive angular margin m is added to minimize the intra-class distances

 - remove the bias term and fix the length of Wj and xi by L2 normalization

   ![Eq 4](Eq/4.PNG?raw=true)

### 2. Comparison

- 특징 벡터와 가중치 사이의 각도가 [0, π]에 속할 때, QAMFace의 target logit 및 logit gradient 곡선은 단조 및 선형

  - 단조 및 선형은 수렴 속도 가속화
  - QAMFace 외 : 비 단조적
  - ArcFace : decision margin의 overlap 존재

  ![Fig 1](Fig/1.PNG?raw=true)

- 예) 동일한 클래스 간 마진(θij = π/6) 공유

  - 오른쪽의 θi가 더 작아 클래스 내 압축성이 더 높음을 나타냄

  - 왼쪽(0.19)의 ArcFace 손실 값이 오른쪽(0.24)보다 작음.

    - 본래 ArcFace의 목적에 의하면 더 높은 패널티를 부여해야 함.
    - 이는 수렴에 적절하지 않음.

    ![Fig 2](Fig/2.PNG?raw=true)

## Experiment

![Table 1](Table/1.PNG?raw=true)

![Table 2](Table/2.PNG?raw=true)

![Table 3](Table/3.PNG?raw=true)

