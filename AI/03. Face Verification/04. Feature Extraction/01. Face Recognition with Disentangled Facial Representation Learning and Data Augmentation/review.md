# **Disentangled Facial Representation Learning and Data Augmentation [1]**

<hr>

[1] : "FACE RECOGNITION WITH DISENTANGLED FACIAL REPRESENTATION LEARNING
AND DATA AUGMENTATION" in *CVPR 2019*



## Abstract

- 얽힌 표현 학습을 위한 Representation-Learning Wasserstein-GAN (RL-WGAN)
- 3D Morphable Model(3DMM)을 활용한 Data augmentation 훈련



## Introduction

- Recognition aims

  - 정면이 아닌 이미지를 정면으로 회전해서 더 나은 얼굴 특징 추출
  - 정면이 아닌 얼굴로부터 포즈 불변 특징 학습
  - 얽힌 얼굴 표현 탐색

- Disentangled Representation learning GAN(DR-GAN)

  - 포즈로부터 face identity를 풀기 위한 G, D를 학습
  - Training issue
    - 수렴의 어려움
    - 몇몇 케이스에서 모드 붕괴 관측
  - 훈련의 어려움을 해결하기 위해 imbalanced training data 이용
    - imbalanced training data는 추정에서 편향된 결과임.
  - imbalanced training data를 위해 data augmentation scheme 적용

  ![1.PNG](Fig/1.PNG?raw=true)



## Proposed approach

- Includes
  - RL-WGAN 수정
    - D로부터 identity, pose classifier를 각각 분리
    - 더 나은 훈련을 위해 Wassertein loss 적용
    - batch norm -> group norm으로 변경
      - group norm
        - 배치의 크기가 극도로 작은 상황에서 BN 대신 사용
        - 각 채널을 N개의 group으로 나누어 일반화 시켜주는 기술
  - Data augmentation module
    - 2D-to-3D face mapping approach



## 3DMM Face Profiling for Data Augmentation

- Face profiling
  - 얼굴 전면의 프로필 뷰를 만들어 참조
  - 3DMM : Multi-Features Fitting(MFF)에 따라 2D면에 맞춰짐
- 주어진 얼굴 이미지에서 facial landmark들을 얻기 위해 Face Alignment Network(FAN), Multi-Dropout Network(MDN) 적용
  - 두 모델에서 감지된 위치의 평균값으로 랜드마크 지정
- 저품질/불필요 이미지 제거 및 랜드마크의 더 나은 localization 포함
- 분산 표현을 포함하기 위해 Basel Face Model(BFM)으로부터 identity shape과 Face Warehouse로부터 expression shape를 결합
- 랜드마크 매칭을 위해 포즈 변화가 있는 얼굴에 3DMM을 맞추고 배경에서 추정된 깊이로 3D meshing

- 3D 메쉬된 면 및 배경 모델은 yaw 및 pitch로 회전하고 렌더링된 2D 이미지

  ![2.PNG](Fig/2.PNG?raw=true)

