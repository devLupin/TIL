# **SphereFace[1]**

<hr>

[1] : SphereFace: Deep Hypersphere Embedding for Face Recognition, in *CVPR 2017*



## Abstract

<hr>

- **Closed-set**
  - Training Set과 Test Set의 분류해야 할 Class가 같은 것

- **Open-set**

  - 서로 다른 class의 feature끼리는 유사도가 낮고, 같은 class의 feature는 유사도가 높도록 학습

  - **Metric Learning** 이라고 함.

    ![Fig 1](Fig/1.PNG?raw=true)

- 얼굴인식에서는 학습할 수 있는 얼굴은 제한되어있지만 실제 응용하기 위해서는 학습에 사용되지 않은 새로운 얼굴도 구별할 수 있는 고유한 feature를 만들 수 있어야 하므로 Metric Learning에 적합
- 이러한 학습을 위해 **A-Softmax Loss** 제안



## Introduction

<hr>

- Pioneering work
  - softmax loss
    - 오직 분리 가능한 특징만 학습 가능
  - some methods combine softmax loss
  - center loss
    - 클래스 내 압축만 명시적으로 권장
  - contrastive loss, triplet loss
    - pair/triplet mining 절차의 민감한 디자인이 요구됨.

- Modified softmax loss
  - 직접적인 각도 최적화
  - CNN이 angularly 분포 특징을 학습하기 위함.
- **A-Softmax loss**
  - 정수 m(m>=1)을 추가하여 결정 경계 제어
  - parameter m을 통해 angular margin의 크기를 제어함.
  - 클래스 간 margin 확대, 클래스 내 각도 분포 압축 동시 진행
  - 학습된 특징은 angular distance metric으로 구성
  - Hypersphere manifold에서 더 효율적인 특징 학습

- 화를 위해 가중치와 바이어스를 각각 1과 0으로 설정 후 내적 공식을 이용해 치환
- 2D feature에서의 비교

![Fig 5](Fig/2.PNG?raw=true)



## Deep Hypersphere Embedding

<hr>

- Softmax Loss

  - x : 학습된 특징 벡터
  - W_i, b_i : class i에 상응하는 last FC layer의 가중치, 바이어스 

  ![Eq 1](Eq/1.PNG?raw=true)

  - 위의 식에서 분모 term은 normalization term이므로 결정하는데 영향이 없으므로 생략

  - 따라서, 결정 경계는 (W1 - W2)x + b1 - b2 = 0인 지점

  - 일반화를 위해 가중치와 바이어스를 각각 1과 0으로 설정하면, θ에만 의존하는 식이 됨.

  - 재정의

    ![Eq 4](Eq/4.PNG?raw=true)

  - 일반화를 위해 가중치와 바이어스를 각각 1과 0으로 설정 후 내적 공식을 이용해 치환

    - 거리 공식으로 각도 공식을 필요로 하기 때문에, 각도 특징이 필요

  ![Eq 5](Eq/5.PNG?raw=true)

- **Angular Margin to Softmax Loss**

  - **cos(mθ1) > cos(θ2)**의 조건을 따른다. 

    - cos(θ1)의 하한은 cos(θ2)보다 커야함.

  - class 1의 결정 경계는 cos(mθ1) = cos(θ2)

    ![Eq 6](Eq/6.PNG?raw=true)

  - 상기 조건을 없애고 CNN에 최적화하기 위해 **cos 함수 -> 단조 감소 함수**로 변환

    ![Eq 7](Eq/7.PNG?raw=true)

  - angular margin은 m이 클수록 증가하고, m=1이면 0

  - A-softmax loss는 Wi =1, bi=0이 요구되고, 이는 샘플과 가중치 사이의 각도에만 의존하게 됨.

  - 샘플 x는 가장 작은 각도를 가지는 identity로 분류

  - 논문에서는 m을 곱한 이유가 기울기 계산과 역전파의 편리함 때문이라고 설명함.

  - 더 큰 m은 각 클래스에 대해 더 작은 Hypersphere 영역으로 이어지지만, 설계가 복잡해짐.

  

  

  ## Comparision

  <hr>

  - Decision boundaries

    ![Table 1](Table/1.PNG?raw=true)

  - Geometry Interpretation 

    - 2D에서의 호의 길이는 3D에서의 지름과 같음.

    ![Fig 1](Fig/3.PNG?raw=true)

