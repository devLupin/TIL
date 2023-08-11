# **DDAT**

<hr>

[1] : "Dual domain adaptive translation for low-resolution face verification in the wild", in *Pattern Recognition 2021*



## Abstract

- end-to-end Low-resolution(LR) face translation, verification 프레임워크 제안
- indentity 일관성, low-level 특성 유지
- 타겟 도메인의 생성된 이미지를 훈련 데이터의 확장으로써 사용



## Introduction

- High-resolution(HR)에서 학습된 모델이 LR에 적용되면, 성능이 상당히 감소할 것
  - HR과 LR 데이터 간 도메인 차이 때문에
- Super-resolution(SR) 모델이 타겟 도메인과 매우 다른 소스 도메인에서 학습되면 타겟 도메인에서의 얼굴 인식률이 좋지 않음.
- 일부 문헌에서 소스 도메인에서 학습된 SR 모델이 잘 수행되지 않는다고 보고
  - 널리 사용되는 SR 기법 중 MSE는 타겟 도메인 이미지에서 high-frequency loss 유도
  - 합성된 LR 이미지와 기존 LR이미지 간의 도메인 차이가 커질 수 있음.
- 도메인 차이를 줄이기 위한 mainstream solution
  - 잠재 공간에서 소스 도메인과 타겟 도메인의 데이터 분포 정렬
  - 그러나, 정렬된 데이터 분포는 identity의 일관성을 유지하지 않고, low-level 특성의 손실을 유도할 수 있음.
- **Dual Domain Adaptive Translation(DDAT)**
  - Adaptive adversarial module을 활용하여 LR 얼굴 이미지에 필요한 정보를 보완하기 위해 고품질 얼굴 이미지 생성
  - 잠재 공간에서 feature-level domain adaptation을 사용하여 소스 도메인, 타겟 도메인의 LR 이미지 분포를 정렬
  - 타겟 도메인의 생성된 이미지를 소스 도메인의 확장으로 처리



## Related works

### 1. General face recognition

- 수작업 특징 또는 학습된 descriptor 이용
- Traditional : LBP, SIFT, HOG, Gabor, etc.
- With deep learing : DeepFace, FaceNet(using triplet loss), Minimum hyperspherical energy(MHE), SphereFace, SphereFace+(+MHE), L-Softmax
  - Angular based : SphereFace, ArcFace, CosFace

### 2. LR Face recognition

- First type : 거리 비교를 위해 이미지와 갤러리 이미지의 특징 맵핑
- Second type : 고품질 이미지를 얻기 위해 LR 이미지의 유용한 정보를 재구성

### 3. DDAT

- 잠재 공간과 데이터 공간에서의 도메인 차이를 동시에 최소화



## Proposed approach

- 합성된 LR 이미지와 기존 LR 이미지 분포 간 수렴

- Dual domain adaptive translation structure(end-to-end)

  - LR face verification acc 향상

- Adaptive adversarial module : image generator, adversarial adaptor, discriminator

  - unseen LR 이미지의 이미지 번역

- Anti perturbation classifier module

  - 훈련 시, 라벨화 되지 않은 타겟 도메인을 추가하여 얼굴 인식 정확도 향상

  ![Fig 2](Fig/2.PNG?raw=true)

### 1. Adaptive adversarial module

- 타겟 도메인에서 HR 얼굴 이미지 생성

  1. Domain adaptive generator

     - G의 backbone으로써 U-Net 적용

     - D를 속이고, 같은 identity 유지

     - Adversarial G의 목적 함수

       - Gan loss + L1 loss

         ![Eq 1](Eq/1.PNG?raw=true)

     - U-Net 병목에서 adversarial adaptor 구현

     - 소스 도메인, 타겟 도메인의 분포 정렬

     - Feature-level domain adaptation object function

       - α and β are weights of source domain labels and target domain labels.
       - A(·) is the output of the adversarial adaptor  .

       ![Eq 4](Eq/4.PNG?raw=true)

  2. Adaptive discriminator

     - 타겟 도메인의 LR 얼굴 이미지는 상응하는 HR 이미지를 갖지 않음.

       - 소스 도메인의 HR 이미지를 사용하여 생성된 이미지 품질 판별

     - G의 최적화 중에, L1 loss를 이용해 identity 유지

     - D의 최적화 중에, 소스 도메인의 HR 이미지와 유사한 합성된 이미지의 분포 생성

     - D의 목적 함수

       ![Eq 5](Eq/5.PNG?raw=true)

  3. Adaptive adversarial module의 목적 함수

     ![Eq 6](Eq/6.PNG?raw=true)

### 2. Anti-perturbation verification module

- 타겟 도메인에서의 얼굴인식 정확도 향상은 생성된 이미지의 품질을 향상하기에 충분하지 않음.

  - center loss, triplet loss 등 현재의 기법을 이용하면 소스 도메인에서는 결과가 좋으나, 타겟 도메인에서는 그렇지 않음.

- 타겟 도메인에서의 정확도 향상

  - identity 유지를 위한 consistency loss 추가

    - 더 나은 생성 및 분류 결과 획득

  - 전체 모델을 더 견고하게 하기 위한 Anti-perturbation loss 추가

    - Virtual adversarial training을 이용해 verification module 일반화

  - center loss 채택

    - classification loss, 각 특징, 특징의 중심과의 거리의 합으로 구성

      ![Eq 8](Eq/8.PNG?raw=true)

  ![Fig 3](Fig/3.PNG?raw=true)

### 3. Implementation details

- LFW 이미지의 degrade
  - Gaussian blur, 4 x down-sampling
    - 소스 도메인의 LR 이미지 생성을 위해 nearest neighbour interpolation 이용
- 소스 도메인 LR 이미지 : 소스 도메인 HR 이미지 : 타겟 도메인 LR 이미지 = 1 : 1 : 1

### 4. Algorithm pseudo-code

![Alg](Alg.PNG?raw=true)



## Experiments

![Table 2](Table/2.PNG?raw=true)

![Table 3](Table/3.PNG?raw=true)

![Table 4](Table/4.PNG?raw=true)



