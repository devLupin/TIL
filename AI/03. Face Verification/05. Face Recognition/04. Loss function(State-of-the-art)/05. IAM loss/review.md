# **IAM Loss[1]**

<hr>

[1] : "Inter-class angular margin loss for face recognition", Signal Processing: Image Communication 2020



## Abstract

- 클래스 간 분산을 증가시키는 것이 목적
- 기존 기법들은 클래스 간 마진을 상수 값으로 제한
- 더 작은 클래스 간 각도에 적응적으로 더 무거운 패널티를 적용하고, 클래스 사이의 각도 마진을 더 크게



## Introduction

- 최근 연구들은 **클래스 간 분산 증가**, **클래스 내 거리 감소**에 초점
- 두 개의 카테고리로 요약
  - 얼굴 특징 간 유클리드 거리 최적화
    - 주로 정규화를 통해 진행
    - ex) Triplet loss
  - 다른 클래스 간 각도 최적화
    - L2-constrained Softmax loss, Ring loss, L-Softmax loss, SphereFace loss, AM-Softmax, NormFace, ArcFace loss
    - 위의 손실 함수들은 유클리드 거리 최적화로써, 추가적인 패널티 부분을 사용하지 않음.
      - '얼굴 특징 간 유클리드 거리 최적화' 부분과 비교
- 최신 연구들은 클래스 간 별개의 각도에 관계없이 마진을 상수 값으로 고정하는 방식 사용
- **inter-class angular margin term을 추가한 새로운 손실 함수 제안**
  - angular loss를 위해 정규화 텀을 추가하는 방식으로 디자인
  - 적응적으로 클래스 간 분산 증가
    - 더 작은 클래스 간 각도에 더 무거운 패널티 부여



## Related work

- 1.Softmax loss

  - i번째 클래스 y에 속하는 j번째 샘플 x의 조건부 확률

    ![Eq 1](Eq/1.PNG?raw=true)

- 2.Inner product 형태로 변경

  - a * b = ||a|| * ||b|| * cos𝜃

    ![Eq 2](Eq/2.PNG?raw=true)

- 3.scaled Softmax loss

  - scale factor s를 이용하여 x, W 일반화 및 그 사이의 코사인 값을 곱함.

  - 네트워크의 수렴을 더 쉽게 만듦.

    ![Eq 3](Eq/3.PNG?raw=true)

- 4.SphereFace

  - scaled softmax loss는 클래스 간 거리를 좁게 하거나 클래스 내 분산을 확장할 수 없음.

  - 각도 𝜃 * m

    ![Eq 4](Eq/4.PNG?raw=true)

- 5.CosFace

  - SphereFace는 독립적으로 수렴이 불가하고, 일반적으로 softmax로 최적화

  - 대안으로, 코사인 값에서 마진을 빼는 방식 사용

    ![Eq 5](Eq/5.PNG?raw=true)

- 6.ArcFace

  - SphereFace, CosFace는 Softmax loss없이 수렴 불가

  - 각도에 마진을 더하는 방식 사용

    ![Eq 6](Eq/6.PNG?raw=true)

- CosFace, ArcFace 모두 클래스 간 각도 최적화를 위해 고정된 마진을 더함.



## Proposed method

- 최신 기법들에 **adaptive penalty term**을 추가한 손실함수

  ![Eq 7](Eq/7.PNG?raw=true)

- i-th 특징 **x**와 j-th 가중치 **W** 간 평균 유사도로 간략하게 정의될 수 있음.

  - 만약 x, w간 각도가 유사하면, numerator(추가된 term)는 커지게 됨.
    - 결과적으로 큰 IAM 손실

- 기존 기법들처럼 상수 m을 제약하지 않음.

- 상기 수식은 log(1-pi)의 형태로 재정의 될 수 있음.

  - 이를 위해 비선형 단조 변환 존재

    ![Eq 12](Eq/12.PNG?raw=true)

- 최종 수식은 음수가 아닌 정규화 파리미터 𝛽를 추가

  - Lbase는 scaled loss

    ![Eq 8](Eq/8.PNG?raw=true)

  - 𝛽를 통해 적응적 가중치를 부여하여 IAM loss와 설정된 loss의 강도를 더 활용

- 각 손실 함수에 대한 분포 비교

  ![Fig 2](Fig/2.PNG?raw=true)

- IAM loss 학습 알고리즘

  ![Alg](Alg.PNG?raw=true)

- optimizer를 이용한 최적화

  ![Alg_Opt](Alg_Opt.PNG?raw=true)



## Differences from other state-of-the-art methods

- AM loss, CosFace
  - inter-class margin 상수값 추가
    - 훈련 시 다른 클래스 간 불일치로 인해 제한적임.
  - IAM은 적응형 마진
- SphereFace
  - angle margin 추가, 훈련 시 Softmax loss 필요
    - 대부분의 학습 스텝에서 softmax loss의 영향이 지배적
- Normface
  - 특징과 가중치 일반화, scale factor s를 추가하여 네트워크의 수렴을 쉽게 만듦.
  - 이는 코사인 유사도를 최적화하고, 클래스 간 마진 향상을 포함하지 않음.
- ACD loss
  - 클래스 간 거리 감소, 클래스 내 마진 증가 기반
  - 올바르게 예측된 샘플을 해당 클래스 중심으로 압축하고, 잘못 분류된 샘플을 예측된 클래스 중심에서 멀리 유지
  - 증가된 클래스 간 마진은 잘못 예측된 샘플이 감소함에 따라 감소함.
  - IAM은 전체 훈련 과정에 영향을 미치고, 평균 클래스 간 마진 측정



## Experiments

- ACC

  ![Table 2](Table/2.PNG?raw=true)

  ![Table 3](Table/3.PNG?raw=true)

  ![Table 4](Table/4.PNG?raw=true)

  ![Table 5](Table/5.PNG?raw=true)

  

