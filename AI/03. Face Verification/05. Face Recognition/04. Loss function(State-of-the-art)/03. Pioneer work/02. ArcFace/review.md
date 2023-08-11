# **ArcFace[1]**

<hr>

[1] : "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", in *CVPR 2019*



## Abstract

<hr>

- 대규모 데이터셋에서의 얼굴 인식을 위해 DCNNs를 이용한 특징 학습의 어려움이 있음.
- Pioneer work
  - Centre loss
    - Euclidean space 에서 feature와 그에 상응하는 중심 class 사이의 거리에 패널티를 주는 방식
  - ShereFace
    - 마지막 FC layer에서 linear transformation matrix 사용
    - feature와 그에 상응하는 가중치 사이의 각에 패널티를 주는 방식
- ArcFace
  - 구 형태에서 측지 거리에 대한 정확한 대응으로 인한 명확한 해석



## Introduction

<hr>

- feature는 클래스 내, 클래스 간 작은 거리를 갖게 됨.
- Pioneer work in classification task
  - softmax loss
    - indentity의 수를 선형 증가 시킴.
    - closed-set classification 문제에서는 분리 가능하지만 open-set 얼굴 인식 문제에서는 충분히 구별되지 않음.
    - target logit curve를 매우 급하게 생성
  - triplet loss
    - face triplet의 수에서의 조합 폭발
    - semi-hard sample mining이 어려움
  - Deep hypersphere embedding
    - multiplicative angular margin penalty 
    - 계산을 위한 근사값이 요구됨.
  - CosFace
    - 직접적으로 target logit에 cosine margin panalty를 더하는 방식

- **ArcFace**

  - 판별 기능 향상 및 안정적인 학습을 목표로 함.

  - Process

    - DCNN feature와 마지막 FC layer 간 내적
      - feature와 가중치의 norm 후에 진행
      - Norm은 L2
    - 현재 feature와 target weight 간 각도를 계산하기 위해 arc-cosine 함수 사용
    - 추가 angular margin을 target angle에 더하고 cosine 함수를 사용하여 target logit을 얻음.
    - 모든 logit에 고정된 feature norm으로부터 re-scale

    ![Fig 2](Fig/2.PNG?raw=true)

  - not need to be combined with other loss functions



## **ArcFace**

<hr>

- softmax loss는 feature embedding을 명시적 최적화 하지 않음.

  ![Eq 1](Eq/1.PNG?raw=true)

- Normalisation step

  - 오직 feature와 가중치 사이의 각도에만 의지

  ![Eq 2](Eq/2.PNG?raw=true)

- Additive angular margin penalty

  - x, W 사이의 angular margin penalty m 추가
  - 8개의 다른 identity로부터 얼굴 이미지 탐색
  - 클래스 내, 클래스 간 판별성 향상

  ![Eq 3](Eq/3.PNG?raw=true)

- Softmax VS ArcFace

  - Softmax : 모호한 결정 경계

  - ArcFace : 가장 가까운 클래스 사이에 더 분명한 격차 적용

    ![Fig 3](Fig/3.PNG?raw=true)

