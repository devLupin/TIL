# **ESRGAN**

<hr>



[1] : "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", in *CVPR 2018*



## Abstract

- 기존 SR-GAN의 생성된 이미지는 원치 않는 불순물이 존재하는 경우가 있음.
- ESRGAN
  - 기본 네트워크 유닛으로써 BN 없는 Residual-in-Residual Dense Block(RRDB) 제안
  - **activation 전**에 이용된 특징으로부터 perceptual loss 향상
  - 밝기의 일관성, 질감 복귀 능력 향상



## Introduction

- perceptual loss는 SR 모델의 최적화를 위해 제안
  - pixel space 대신에 feature space에서
  - 더 자연스러운 이미지처럼 보임.
  - 복구된 texture detail 향상
- 기본 모델은 residual block들로 설계되고, GAN 프레임워크에서 perceptual loss를 이용하여 최적화
  - 그러나, 여전히 SRGAN의 결과와 GT 사이에 명백한 차이 존재
- ESRGAN
  - RDDB 제안
    - 더 높은 캐퍼시티 및 쉬운 학습
  - BN 제거 및 residual scaling
  - RaGAN을 이용하여 D 향상
  - activation 전에 VGG 특징을 이용하여 perceptual loss 향상



## Proposed Methods

### 1. Network Architecture

- SRGAN의 복구된 이미지 퀄리티 향상을 위해 G의 두 가지 수정

  - 모든 BN layer 제거

    - 훈련 통계와 테스트 데이터가 많이 다른 경우, BN layer는 불순물을 포함하고, G의 능력을 제한 시키게 됨.
    - 안정된 학습, 일관된 성능
    - 향상된 생성 능력, 계산 복잡도 및 메모리 사용량 감소

  - 기존 블럭을 제안된 RRDB로 변경

    - Multi-level residual network

    - dense block 사용

      - 네트워크 캐퍼시티가 dense connection으로 인해 더 높아짐.
        - Dense connection
          - Dense block 내 한 layer의 input feature를 이후 layer의 input에 concatenate

      ![Fig 4](Fig/4.PNG?raw=true)

- 매우 깊은 네트워크에서의 학습을 위한 추가 기술

  - Residual scaling
    - 불안정 방지를 위해 main path에 추가하기 전에 0~1 사이의 상수를 곱하여 residual 축소
  - Smaller initialization
    - 초기 파라미터 분산이 작아질 때 더 쉬운 학습 가능

### 2.  Relativistic Discriminator

- Relativistic GAN 기반 D(RaD) 사용

- 기존 SRGAN의 D는 input과 real에 대한 확률 추정

- Relativistic GAN의 D는 real이 fake보다 얼마나 더 realistic한지 확률 예측

- 기존 D를 RaD로 교체

  ![Fig 5](Fig/5.PNG?raw=true)

- D loss fomular

  - E는 미니 배치에서 모든 fake 데이터의 평균을 취하는 operation

    ![Eq 1](Eq/1.PNG?raw=true)

- G의 adversarial loss

  - 생성된 이미지, real data 모두의 기울기를 취할 수 있음.

    ![Eq 2](Eq/2.PNG?raw=true)

### 3. Perceptual Loss

- activation 이전에 특징을 제약하여 더 효율적으로 구현

- pre-trained deep network의 activation layer 이전에 정의하여 단점 해결

  - 매우 깊은 네트워크에서 활성화 된 특징은 매우 희소

    - weak supervision. 즉, 성능 감소

      ![Fig 6](Fig/6.PNG?raw=true)

  - GT와 비교하여 재 구성된 이미지의 밝기가 유사하지 않음.

- G의 total loss

  - L1 : recovered image, GT 간 1-norm 거리를 검증하는 content loss

  - λ, η : 서로 다른 loss의 균형을 위한 계수

    ![Eq 3](Eq/3.PNG?raw=true)

- 더 suitable한 perceptual loss 구현

  - material recognition 을 위한 fine-tuned VGG network 기반
  - object 보다 texture를 중점으로

### 4. Network Interpolation

- GAN 기반 기법에서 노이즈 제거

  1. PSNR(Peak Signal-to Noise Ratio)-oriented network 학습
  2. GAN-based network fine-tuning
  3. 1, 2의 상응하는 모든 파라미터 보간

- Interpolated model

  - θ : 각각의 파라미터

  - α : [0, 1] 사이의 interpolation 파라미터

    ![Eq 4](Eq/4.PNG?raw=true)

- 장점

  - 추가 조작 없이 어떠한 α에 대한 의미있는 결과 생성
  - 균형있는 perceptual 퀄리티
  - 모델의 재훈련없이 fidelity 향상



## Result

![Fig 1](Fig/1.PNG?raw=true)

