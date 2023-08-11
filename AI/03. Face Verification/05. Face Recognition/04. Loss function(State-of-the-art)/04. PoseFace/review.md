# **PoseFace**

<hr>

[1] : "PoseFace: Pose-Invariant Features and Pose-Adaptive Loss for Face Recognition" in CVPR 2021



## Abstract

- 제한되지 않은 환경(감시 등)에서 포즈 변화가 큰 경우 성능 저하
  - identity 정보는 포즈 전반에 걸쳐 일관되어야 함.
  - 얼굴 이미지, 프로파일 이미지 간의 데이터 불균형을 인식하지 않아야 함.
- **랜드마크를 활용하여 포즈 불변 특징의 얽힘을 풀고, 포즈 적응 손실 함수를 활용하여 데이터 불균형 문제를 적응적으로 해결**



## Introduction

- 대부분의 얼굴 인식 모델 성능 감소

  - 정면 이미지와 프로파일 이미지 데이터 불균형이 심함.
  - 프로필 얼굴 인식의 어려움

- 다양한 포즈에서의 인식

  - 프로필 얼굴을 정면 뷰로 일반화
    - 추가 계산, identity 정보 손실
  - 포즈 불변 얼굴 특집 학습
    - 다중 경로를 포즈 인식 변환으로 통합
    - 얽힌 표현 존재
    - 다양한 실세계 데이터셋에 적용할 수 없음.

- 해결책

  - **종단 간 및 얽힌 표현을 자체적으로 푸는 방식으로 포즈 불변 임베딩**

    - 각 입력 얼굴을 포즈와 identity 특징으로 동시에 매핑

      ![Fig 1](Fig/1.PNG?raw=true)

    - 두 특징 subspace 간 선형, 직교 강제화

    - identity 정보가 포즈 feature로 얽히지 않음.

    - AutoEncoder 디자인

      - 사전 학습 모델
      - 얼굴 랜드마크, 특징벡터 1:1 매핑
        - 특징 벡터는 랜드마크에만 관련됨.

    - 풀림으로 인한 인식 성능 저하 방지

  - **Pose-adaptive loss**

    - 프로파일, 정면 얼굴 간 데이터 불균형 제어
    - 프로파일 얼굴의 비율이 더 적음.
      - Euler 각으로 구성 > 60도
    - weight 손실을 낮춤.
      - ArcFace loss 재구성
      - hard sample에 초정을 맞춘 학습



## Related Works

### 1. Pose-Invariant Feature Representation

- c-CNN
  - 각 레이어의 커널은 특징 표현에서 희소하게 활성화
- PAMs
  - 여러 포즈별 모델의 점수를 융합하여 포즈 변화 처리
- p-CNN
  - identity 분류를 주작업으로 그 외는 부작업으로 진행
- DFN
  - 특징 정렬
  - 같은 identity의 샘플 쌍, 클래스 간 특징 분산 감소를 위한 패널티 적용
- DREAM
  - 프로파일 얼굴 표현을 canonical frontal pose로 변환
- 상기 모델의 경우, 완벽하게 풀린 표현을 달성할 수 없었고, 데이터 불균형 이슈 또한 해결 못함.
- **프로파일 얼굴 비율이 적은 경우에도 데이터셋에 직접 적용**

### 2. Face Frontalization

- 얼굴을 canonical frontal view로 일반화
- 얼굴 합성 사용
- 최근에는 GAN 모델을 사용
  - 실제 같지 않은 질감
  - 합성된 이미지 identity 정보 손실
  - 높은 계산 비용

### 3. Disentanglement in Face Recognition

- Disentangling factors : pose, age, expressions

- Methods

  - 특징 크기로부터 나이 회귀
    - 크기는 스칼라이므로 복잡한 factor를 처리하기에 불안정
  - 같은 identity의 얼굴 간 차이 패널티
    - 다른 factor 또는 일관된 분포 필요
  - 이들 모두 같은 identity의 pair-wise 입력 필요, 제한적 사용, 정면 얼굴이 매우 불균형

- **직교 제약에 의해 서로 관련 없는 identity, 랜드마크 공간에 얼굴 매핑**

- **pair-wise input, 추가 레이블 필요 없음.**

  ![Table 3](Table/3.PNG?raw=true)



## Methodology

- 주된 관측
  - 얼굴은 identity 정보, variation 정보로 구성되고 이들은 관계가 없음.
  - head pose 정보는 얼굴 랜드마크에 의해 인코딩 될 수 있음.
- 일반적으로, 2D, 3D 얼굴 랜드마크는 Head pose를 완전히 표현하는데 필요
  - 2D 얼굴 랜드마크는 같은 면에 분포되어 있으므로, 3D head pose 표현에는 부적합
- 본 논문에서는 **2D 이미지에서 self-disentanglement에 의해 포즈 관련 정보 제거**
  - 2D 얼굴 랜드마크를 도입해 포즈 정보 필터링을 도울 수 있음.

### PoseFace module

- Landmark module

  - 사전학습 모델
  - 포즈 특징 추출
  - 훈련 단계에서, identity module과 pose module 학습
  - self-disentanglement 달성

- Adaptive recognition loss

  - prevailing ArcFace 수정
  - 프로파일 얼굴 가중치 증가

- testing 단계에서 identity 모듈은 ArcFace와 같음.

  ![Fig 2](Fig/2.PNG?raw=true)



## Pose-Adaptive ArcFace (PAA) Loss

- 인식의 어려움, frontal and profile 샘플의 불균형 개선

- 얼굴 각도 기반 margin을 적용하여 ArcFace loss 수정

  - mb : base margin
  - δm : additional margin
  - ri : added margin의 총량 제어, 0 ~ 1 사이의 값

  ![Fig 4](Fig/4.PNG?raw=true)

- Formulation

  - Euler 각에 따라 계산

  - identity feature classification loss

    ![Eq 2](Eq/2.PNG?raw=true)



## PAA Loss vs ArcFace Loss

- ArcFace Loss

  ![Eq 2-1](Eq/2-1.PNG?raw=true)



## ACC

- In CPLFW benchmark under the 10-fold protocol

  ![Table 6](Table/6.PNG?raw=true)

- Verification evaluation (%) according to different FARs on IJB-B and IJB-C

  ![Table 7](Table/7.PNG?raw=true)