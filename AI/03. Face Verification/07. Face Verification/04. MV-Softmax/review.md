# **MV-Softmax**

<hr>

[1] : "Mis-classified Vector Guided Softmax Loss for Face Recognition", in *CVPR 2019*



## Abstract

- Margin-based softmax loss
  - 다른 클래스들 간 feature margin 증가
  - feature 정보의 중요성 무시
  - non GT에 대한 판별 없음.
- MV-Softmax
  - 적응적으로 mis-classified feature vector를 강조
  - margin 및 feature mining의 장점 상속



## Introduction

- SV-Softmax
  - 대규모 데이터셋(MegaFace Challenge, Trillion-Pairs Challenge 등)에서 매우 낮은 FAR 기록
- 그 외 최근 연구들
  - 기본적인 이론은 클래스 내 축소 및 클래스 간 분리가 최대화 되면 판별력 높아짐.
  - Angular margin(A-Softmax)
    - GT, 다른 클래스 사이의  분산을 더 크게 만듦.
    - 그러나, 때때로 불안정하고, 최적의 파라미터를 요구함.
  - Additive margin(AM-Softmax)
    - 최적화의 안정성을 높이기 위함.
  - Additive angular margin(Arc-Softmax)
    - 명확한 기하학적 해석
- 최근 연구들의 단점
  - 판별력 학습을 위한 특징 정보의 중요성 무시
    - mining-based softmax loss에 의존하게 될 수 있음.
    - Hard sample은 경험적으로 결정되고, easy sample은 완전히 무시
    - Hard sample의 표현이 불명확함.
  - 오직 GT에 근거하여 feature margin을 확대함.
  - 같거나 고정된 feature margin을 모든 클래스에 대해 적용
- MV-Softmax Contiribute
  - **Hard sample을 잘못된 벡터로 명시적으로 표시하고 이를 강조하여 판별 특징 학습**
  - **다른 non GT 클래스로부터 판별 가능성을 흡수하고, 적응형 margin 사용**
  - **Feature margin, feature mining 기술의 장점만을 상속**



## Preliminary Knowledge

- Softmax

  ![Eq 1](Eq/1.PNG?raw=true)

- Mining-based Softmax

  - 유익한 examples에 초점을 맞춘 학습

  - loss value[g(Py)] 기반

    ![Eq 2](Eq/2.PNG?raw=true)

    ![Eq 2-1](Eq/2-1.PNG?raw=true)

    ![Eq 2-2](Eq/2-2.PNG?raw=true)

- Margin-based Softmax

  - 기본적으로 cos, f는 동일

  - A-Softmax, AM-Softmax, Arc-Softmax마다 threshold가 달라짐.

    - 오직 ground truth class y의 관점

      ![Eq 3](Eq/3.PNG?raw=true)

      

## Problem Formulation

- Disadvantages of Mining, Margin based
  - 판별 학습을 위한 feature 정보 중요성 무시
  - 다른 non GT 클래스로부터 잠재적인 식별 가능성을 인식하지 않음.
  - 단순히 동일하고 고정된 margin m1, m2 또는 m3를 사용하여 다른 클래스 간 feature margin 확대

### 1. Naive Mining-Margin Softmax Loss

- L2 + L3

- 여전히 ground truth class에만 초점이 맞춰짐.

- 여전히 경험적으로 선택

- hard example에서의 정의가 모호함.

  ![Eq 4](Eq/4.PNG?raw=true)

### 2.  Mis-classified Vector Guided Softmax Loss

- mis-classified feature는 판별 능력 향상에 있어 더 중요하다는 개념

- 유익한 feature 중심 학습

- mis-classified feature는 1로 지정되어 일시적으로 강조

  ![Eq 5](Eq/5.PNG?raw=true)

- 최종 loss

  ![Eq 6](Eq/6.PNG?raw=true)

  - h(...) > 1, 표기된 mis-classified vector를 강조하기 위한 re-weight function

  - h(...)의 두 가지 방식

    - 잘못 분류된 모든 클래스에 대해 고정 가중치를 갖게 함.

      ![Eq 7](Eq/7.PNG?raw=true)

    - adaptive formulation

      ![Eq 8](Eq/8.PNG?raw=true)

  - t >= 0, 사전 설정된 하이퍼파라미터

  - t = 0, margin-based softmax loss와 같아짐.

### #. SV-Softmax vs MV-Softmax

- t : 사전 설정된 하이퍼파라미터

- SV-Softmax h() function

  ![Eq 8-1](Eq/8-1.PNG?raw=true)

- MV-Softmax h() function

  ![Eq 8](Eq/8.PNG?raw=true)

- 논문에서는 SV-Softmax가 대규모 데이터셋에서 매우 낮은 FAR을 기록했다고 언급
  - 그 외에 사항은 언급되지 않음.