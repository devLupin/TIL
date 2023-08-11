# **SV-Softmax** [1]

<hr>

[1] : "Support Vector Guided Softmax Loss for Face Recognition", in *CVPR 2018*



## Abstract

- CNN은 특징 판별 한계가 드러나기 시작함.
- 이를 해결하기 위해, 두 가지 파트로 나뉨.
  - Mining based strategies
    - 유익한 것에 초점을 두는 방식
  - Margin based loss function
    - 실제 값의 관점으로부터 특징의 margin 증가
- 상기 두 방법은 거대한 샘플들에서 모호성이 존재하거나, 판별 능력 부족
- 잘못 분류된 점(SV)을 적응적으로 강조하는 기법 제안



## Introduction

- 아래부터 희소 데이터, 거대 데이터에서의 판별성 향상을 위한 연구에 대해 설명
- Metric learning loss function
  - contrastive loss, triplet loss
  - 높은 계산적 비용을 감수해야 함.
- Mining based loss function
  - HM-Softmax : high-loss example을 이용한 미니배치 구성으로 성능 향상
  - Focal loss : 모델 복잡도 기반 앙상블 학습
- Margin-based loss function
  - hard sample의 최적화가 아닌, 다른 클래스 간 feature margin 증가하는 기법
  - intra-class의 축소 능력 향상을 위해 각 identity의 중심 학습하는 기법
  - scale 파라미터를 사용하여 높은 기울기를 만들어 softmax loss 제어하는 기법
  - A-Softmax : 더 큰 클래스 간 분산을 위해 다른 클래스와 실제 클래스 간 angular margin을 구하는 기법
    - 상기 언급된 방법들은 unstable하고, 최적의 파라미터를 찾는 데 어려움 존재
  - AM-Softmax : additive margin 디자인
  - Arc-Softmax : additive angular margin 디자인
- Disadvantage
  - Mining based
    - hard example에서의 정의가 모호하여 때때로 경험적으로 선택해야 함.
  - Margin based
    - 실제 클래스의 관점에서만 feature margin을 확대하여 해석(self-motivation)
    - 실제가 아닌 클래스에 대한 학습 결과는 때때로 무시됨.
- **Main contribution**
  - hard sample의 모호성 제거
  - SV에 초점을 맞춰 다른 클래스의 판별력 향상
  - Mining, Margin based 기법을 의미적으로 융합



## Preliminary Knowledge

- Softmax

  ![Eq 1](Eq/1.PNG?raw=true)

- Mining-based Softmax

  - 유익한 examples에 초점을 맞춘 학습

  - loss value[g(Py)] 기반

    ![Eq 2](Eq/2.PNG?raw=true)

    ![Eq 2-1](Eq/2-1.PNG?raw=true)

    ![Eq 2-2](Eq/2-2.PNG?raw=true)

  - Py : ground truth 확률

  - g : Focal loss

- Margin-based Softmax

  - 기본적으로 cos, f는 동일

  - A-Softmax, AM-Softmax, Arc-Softmax마다 threshold가 달라짐.

    - 오직 ground truth class y의 관점

    ![Eq 3](Eq/3.PNG?raw=true)





## Formulation

<hr>

### 1. Naive Mining-Margin Softmax Loss

- L2 + L3

- 여전히 ground truth class에만 초점이 맞춰짐.

- 여전히 경험적으로 선택

- hard example에서의 정의가 모호함.

  ![Eq 4](Eq/4.PNG?raw=true)

### 2. Support Vector Guided Softmax Loss

- mis-classified feature는 판별 능력 향상에 있어 더 중요하다는 개념

- 유익한 feature 중심 학습

- mis-classified feature는 1로 지정되어 일시적으로 강조

  ![Eq 5](Eq/5.PNG?raw=true)

- hard example이 명확하게 정의되고, SV의 희소 집합에 초점을 둠.

- SV-Softmax

  - t : 사전 설정된 하이퍼파라미터

  - h() : indicator function

  - if t=1... softmax와 동일

    ![Eq 6](Eq/6.PNG?raw=true)

    ![Eq 7](Eq/7.PNG?raw=true)

- **Binary mask function, I() 와 Indicator function, h()를 Softmax에 추가하여 Naive Mining-Margin Softmax loss의 단점 해결.** 

- Example

  - cos(wy, x) : ground truth와의 각도
  - cos(wk, x) : non-ground truth와의 각도
  - non-sup vector
    - 샘플과 정답 간 각도가 샘플과 오답 간 각도보다 적음. (좌변이 우변보다 큼)
    - 샘플이 오답보다 정답에 가까움.
    - 잘 분류된 상태이므로 loss에 weight를 높이지 않음.
  - sup vector
    - 좌변이 우변보다 적음.
    - 샘플이 정답보다 오답에 가까움.
    - 잘못 분류하고 있는 상태이므로, hard sample로 인지하고, loss에 weight를 높임.
      - 기존 cos에 h를 곱함.

  ![Fig 1](Fig/1.PNG?raw=true)

### 2.1 Releation to Mining-based Softmax Losses

- 아래는 모두 binary classification 예시

- Focal loss

  - hard example에 대한 강조

  - loss 관점에서 직접적이고 hard example의 정의가 모호함.

    ![Eq 8](Eq/8.PNG?raw=true)

- SV-Softmax

  - 결정 경계에 따라 hard sample(SV) 정의

  - SV x1에 대한 확률 감소

    ![Eq 9](Eq/9.PNG?raw=true)

- Example(Binary cases)

  - SV-Softmax
    - loss -log(Py)에서 확률 값을 조정함으로써 loss를 높임.
  - Focal loss-Softmax
    - 확률은 변화하지 않고 loss 자체를 변경하여 hard sample loss의 비중을 높임.

  ![Fig 2](Fig/2.PNG?raw=true)

### 2.2 Relation to Margin-based Softmax Losses

- Margin-based

  - 정답에 가깝게 학습

  ![Eq 10](Eq/10.PNG?raw=true)

- SV-Softmax

  - non-ground truth class의 관점에서 feature margin 증가

  - 오답으로부터 멀리 되도록 학습

  - theta 1은 ground truth, theta 2는 non

    ![Eq 11](Eq/11.PNG?raw=true)

- SV-Softmax loss vs Margin based

  - SV-Softmax는 non-ground truth class의 관점으로,

  ![Fig 4](Fig/4.PNG?raw=true)

### 2.3 Pipeline of losses

- SV-Softmax = Margin based, Mining-based의 역할을 one framework로

- VS margin based

  - Margin based는 모든 Ground truth의 값을 변경
  - SV-Softmax는 일부 값만 변경
    - Loss의 binary mask에 의해, **정답이 아닌 클래스 중 hard sample로 분류될 만큼 오답에 가깝지 않으면, 값을 바꾸지 않음.**
    - 즉, cos(wy, x) > cos(wk, x) 인 경우

- VS Mining based

  - Mining based는 마지막 계산에서의 Loss 자체를 변경
  - SV-Softmax는 Softmax를 거친 이후 output 변경

  ![Fig 3](Fig/3.PNG?raw=true)

### 2.4 SV-X-Softmax

- 기존의 SV-Softmax는 오답에 집중하여 Mining based + Margin based를 구현

- F function을 추가함으로써 정답에 가까워지도록 하는 효과 구현

  - 정답 class에는 가깝도록, 오답 class에는 멀도록

- 최종 손실 함수

  ![Eq 12](Eq/12.PNG?raw=true)

  - margin based 결정경계에 의해 Indicator mask(Ik)가 다시 계산됨.

    ![Eq 13](Eq/13.PNG?raw=true)

- SV-Softmax vs SV-X-Softmax  

  ![Fig 5](Fig/5.PNG?raw=true)



## Optimization

- SGD를 위해 편미분을 해봤더니, **W**에 대한 편미분일 때 x term, **x**에 대한 편미분일 때 W term이 생기는 것 말고는 기존 Softmax와 동일

  - **SGD로 학습 가능**

  ![Eq 14](Eq/14.PNG?raw=true)



## Algorithm

- Loss는 CNN을 통해 feature map을 다시 softmax를 거쳐 학습함.
- theta : CNN의 weight
- **W** : head 부분 weight

![alg](alg.PNG?raw=true)



## ACC

- LFW test data

  ![Table 1](Table/1.PNG?raw=true)

- MegaFace Chellenge

  ![Table 2](Table/2.PNG?raw=true)

- Trillion Pairs Challenge

  ![Table 3](Table/3.PNG?raw=true)

  