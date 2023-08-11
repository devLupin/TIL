# **EqM Loss**

<hr>

[1] : "An Equalized Margin Loss for Face Reconition", in *IEEE 2020*



## Abstract

- inter-class margin과 intra-class scope를 모든 클래스에 걸쳐 유사하게 만듦.
- hard sample mining을 활용하여 inter-class simlilarity 하한과 균등한 마진 보장
  - intra-class similarity의 상한 또한 제어



## Introduction

- Face recognition의 시나리오
  1. closed-set
     - GT label과 predict label 비교
  2. open-set
     - test 데이터가 없는 경우
- 논문에서는 open-set face recognition을 위해 loss function 제안
  - Enhancing intra-class similarity, Reducing inter-class similarity
- Pioneer work
  - Metric learning
    - reducing intra-class distance, enlarging inter-class distance
      - 특징의 거리 최적화를 통해
    - Euclidean distance, Cosine distance 등
    - 제한된 학습을 위해 적합한 얼굴 이미지를 선택하는 방법에 크게 의존하는 경향
    - 다른 기법들과 대비해 효율성이 떨어짐.
  - Hard-sample mining
    - hard sample에 더 무거운 가중치를 부여하여 집중시키는 방식
  - Margin enlarging(Margin-based)
    - 특징 간 각도 최적화, 더 판별적인 특징을 얻기 위해 클래스 간 상수 마진을 더함.
    - SphereFace, NormFace, etc,.
    - 클래스 간 마진을 제약할 뿐, intra-class 불일치를 고려하지 않음.
    - 특정 클래스에 관계없이 고정된 상수 마진 적용
  - Data-imbalance mitigating
    - 많은 이미지를 가진 인원에 대한 편향됨을 회피
- EqM loss
  - Margin based method : inter-class distance 제약
  - Hard-sample mining : small intra-class scope로 만들어 데이터 불균형 문제 해결
  - 모든 클래스의 특징 분포를 고르게 하고, 데이터 불균형의 부정한 효과를 완화



## Related work

- Margin-based Method

  - C: number of class

  - Xi : i-th nomalized sample feature

  - Yi : label

  - W : normalized weights

    ![Eq 1](Eq/1.PNG?raw=true)

- **Margin-based Methods**

  - (2) : Angular softmax loss(A-Softmax)

    - 다른 결정 경계 생성을 위해 하이퍼 파라미터 m을 더함.

  - (3) : GA-Softmax

  - (4) : ArcFace

    - 더 나은 수렴을 위해 클래스 간 고정된 angular margin m을 더함.

  - (5) : Additive Margin loss(AM loss)

    - 클래스 간 고정된 cosine margin을 더함.

    ![Eq 2-5](Eq/2-5.PNG?raw=true)

  - A-Softmax만 x의 L2-norm을 이용해 scale parameter를 조정하고, 나머지는 고정된 상수 scale parameter 사용

  - inter-class distance를 키우는 데 초점이 되어 있고, 클래스 특징의 차이는 무시됨.

- **Hard-Sample Mining Methods**

  - 전통적인 focal loss

    - γ : easy sample, hard sample의 상대적 중요성 조절을 위한 하이퍼 파라미터

    ![Eq 6](Eq/6.PNG?raw=true)

    ![Eq 6-1](Eq/6-1.PNG?raw=true)

  - 네트워크의 기능 촉진

  - hard sample은 intra-calss distance와 inter-class variance 모두 고려되어 정의

- **Imbalanced Data Problem**

  - 전통적인 기법
    - 미니배치에서 reducing intra-personal variance, enlarging inter-personal difference를 통해 완화하는 방법
    - minority classes(분포가 적은)에 좀 더 집중하는 방법



## Proposed method

### A. EqM Loss

- 손실함수 정의

  - cos θi,yi  : intra-class similarity
  - cos θi,j  : inter-class similarity
  - t1 : intra-class similarity의 하한
  - t2 : inter-class similarity의 상한
  - t1-t2 : offset term

  ![Eq 7](Eq/7.PNG?raw=true)

- intra-class similarity가 커질수록 더 compact한 결과 확보 가능

  - cos θi,j가 가능한 한 작아야 큰 클래스 간 마진

- 4가지 경우

  1. inter-class margin이 커져 θ가 0에 가까워지는 경우

     ![Eq 8](Eq/8.PNG?raw=true)

  2. 가장 이상적인 결과

     ![Eq 9](Eq/9.PNG?raw=true)

  3. 최악의 경우

     - inter-class margin을 증가시키고, intra-class의 축소

     ![Eq 10](Eq/10.PNG?raw=true)

  4. inter-class similarity이 증가하여 θ가 0에 가까워지는 경우

     ![Eq 11](Eq/11.PNG?raw=true)

- 두 파라미터(t1, t2)는 inter-class scope, intra-class scope의 동등한 제어, 모든 클래스에 편향되지 않은 결과

  - 데이터 불균형 문제 완화



### B. EqM Loss의 장점

- Easy sample, Hard sample

  - 과도한 패널티는 가중치를 hard sample에 지나치게 편향되게 하므로 처벌의 정도는 적당해야 함.

  - hard sample에 더 무거운 패널티 부여 예시

    - yello dots, green dots : easy samples, hard samples
    - blue line, red line : class y, decision boundary

    ![Fig 3](Fig/3.PNG?raw=true)

  - if cos θi,yj < t1, 더 무거운 패널티 부여

  - if cos θi,yj > t1, easy sample

- **Imbalanced Data**

  - 예시 : AM-loss for CASIA-WebFace

    - 분포가 적은 샘플에 따른 정확도 실험 결과

    - 분포가 80개 미만(heavy head라고 칭함.)인 샘플들만 벤치마크한 결과

      ![Fig 5](Fig/5.PNG?raw=true)

  - 기존

    - 즉, 더 큰 inter-class margin
    - AM Loss는 대다수 클래스에 대한 inter-class 범위를 제한하지 않으면서 안정적인 intra-class margin만 제한하기 때문
    - 다수 클래스의 넓은 범위는 inter class margin 확장을 방해하고, 불균형 공간 분포로 이어짐.

  - EqM loss

    - inter-class scope, inter-class scope 모두 제약
      - 더 좋은 분포로 이어짐.

    ![Fig 1](Fig/1.PNG?raw=true)

    - 데이터 불균형과 분류된 샘플(hard sample, easy sample)의 테스트 정확도 비교

      - 좀 더 일정한 정확도

      ![Fig 6](Fig/6.PNG?raw=true)

    - LFW에서의 분류된 샘플과 정확도 비교

      ![Table 1](Table/1.PNG?raw=true)

- **Optimization**

  - 기울기가 0이 아닌 특정 간격에서 매개변수를 최적화하기만 하면 됨.

  - 가중치 wyi는 클래스 내 압축성이 임계값 t1보다 낮을 때 업데이트되고, wj의 업데이트는 클래스 간 유사도가 임계값 t2보다 높을 때만 필요

  - t1은 클래스 내 유사도의 하한 결정, t2는 클래스 간 유사도의 상한 결정

  - 증명 수식

    ![Eq 13-15](Eq/13-15.PNG?raw=true)

    ![Eq 16-18](Eq/16-18.PNG?raw=true)



## ACC

![Table 2](Table/2.PNG?raw=true)

![Table 3](Table/3.PNG?raw=true)

![Table 4](Table/4.PNG?raw=true)

