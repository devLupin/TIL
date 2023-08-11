# **VarGFaceNet** [1]

<hr>
[1] : "VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition" in *CVPR 2019*




## Abstract

- variable group convolutional network 제안
- 네트워크 시작 부분에 필수 정보를 받고, 임베딩을 위한 FC layer의 파라미터 감소를 위한 특정 임베딩 셋팅 제안
- 경량 네트워크를 위한 angular distillation loss 제안
- 반복적 지식 증류를 사용하여 교사 모델, 학생 모델 사이의 불일치 해소



## Introduction

- 스마트폰과 같은 임베디드 시스템에서 얼굴 인식은 계산 비용의 제한적인 문제

  - 분류되기 위해서 large scale identity가 필요하기 때문

- 계산 비용 감소를 위한 사전 연구 모델

  - SqueezeNet(SE)

    - 1 x 1 conv
    - AlexNet보다 50배 적은 파라미터 달성

  - MobileNet

    - depthwise separable conv
    - latency, accuracy 사이의 trade off 달성

  - MobileNet V2

    - 거꾸로 된 병목 구조
    - 판별성 향상

  - ShuffleNet, ShuffleNet V2

    - pointwise group convolution, channel shuffle operation 사용

      - pointwise convolution

        - 채널 방향의 conv 진행
        - 채널 축소 시에 주로 사용
        - 입력 영상에 대한 Spatial feature는 추출하지 않은 상태로, 각 채널에 대한 연산만 수행

        ![Fig 1](Fig/pointwise conv.png?raw=true)

    - 계산 비용 감소

- **그러나, 여전히 최적화 문제 존재**

- VarGNet

  - variable group convolution 제안
    - 블록 내 계산 강도의 불균형 해결
    - 더 큰 capacity 가짐.
      - depth wise convolution과 같은 kernel size
  - 더 필수적인 정보를 추출하는데 도움
  - 메모리와 계산 비용을 절약하기 위해 spatial area를 절반으로 감소
  - **그러나, 얼굴 인식에 적합하지 않음.**
    - 임베딩의 마지막 conv, FC layer에만 average pooling layer
    - 이는 **충분히 차별적인 정보를 추출하지 못함.**

- **VarGFaceNet**

  - Based on VarGNet
  -  Add Squeeze-and-excitation network block
    - Channel-wise feature response를 적절하게 조정
    - feature map에 대한 전체 정보를 요약하는 Squeeze operation + feature map의 중요도를 스케일해주는 excitation operation
  - Add PReLu
    - Parametric ReLu
    - 음수에 대한 gradient를 변수로 두고 학습을 통하여 업데이트
  - 네트워크 시작 부분의 downsample 프로세서 제거
  - variable group convolution 적용
    - FC layer 이전에 feature tensor를 1 x 1 x 512로 축소
  - 훈련 중 knowledge distillation 적용
  - angular distillation loss와 동등한 손실 함수 적용
  - **recursive knowledge distillation**
    - 한 세대에서 훈련된 학생 모델을 다음 세대를 위한 사전 훈련된 모델로 취급

- **Contributions are summarized**

  - 필수적인 정보를 전달받기 위해 1 x 1 conv layer를 통해 1024개의 채널로 확장
  - 계산 비용 절감을 위한 variable group conv, pointwise conv 사용
  - recursive knowledge distillation 제안



## Variable Group Convolution

- 채널 수 상수 S로 지정

  - 논문에서는 8로 지정

- 상수 채널 수는  conv에서 variable 수의 그룹 n_i로 이어짐.

- variable group convolution의 계산 비용은 아래와 같음.

  ![Eq 1](Eq/1.PNG?raw=true)

- 블룩 내에서 더 균형잡힌 계산 비용을 위해 depthwise conv 대신 pointwise conv 사용

-  S > 1의 조건 만족(vs depthwise conv)

  - 더 높은 MAdds와 더 큰 네트워크 capacity
  - 더 많은 정보 추출 가능



## Blocks of Variable Group Network

- VarGNet

  - input channel의 수 = output channel의 수

- VarGNet에서 블록 시작의 **채널 수 C를 2배로 확장**

  - block의 generalization 능력 유지

- normal block in VarGFaceNet

  ![Fig 1-a](Fig/1-a.PNG?raw=true)

- down sampling block in VarGFaceNet

  ![Fig 1-b](Fig/1-b.PNG?raw=true)

- normal block에 SE block 추가

- ReLU 대신 PReLu 적용

  - 블록의 판별 능력 향상



## Lightweight Network for FR

### 1. Head setting

- 네트워크의 시작에서 3 x 3 Conv, stride 1 사용

  - 기존 3 x 3 conv, stride 2

- VarGNet에서 첫 번째 conv의 output feature 크기는 다운샘플링되지만 본 모델은 입력 크기와 동일하게 유지

  ![Fig 1-c](Fig/1-c.PNG?raw=true)

### 2. Embedding setting

- 기존 연구에서의 변화
  - 본 연구에서는 feature map의 크기를 1 x 1 x 512로 축소
  - 마지막 conv의 output feature 연관성이 커질 때 FC layer의 파라미터가 거대해지는 현상 발생
- 1 x 1 x 512 feature tensor는 제한적인 feature를 포함하기 때문에 임베딩이 안전하지 않음.
  - variable group convolution + pointwise convolution 추가
  - 파라미터, 계산 비용 감소
- 채널 수를 320에서 1024로 변환하기 위해 1 x 1 conv 사용
-  7 × 7 variable group convolution layer  추가
  - feature tensor를 7 x 7 x 1024 에서 1 x 1 x 1024로
- pointwise convolution은 채널을 연결하고 기능 텐서를 1 × 1 × 512로 출력
- 결과적으로 **30M개의 파라미터를 5.78개의 파라미터로 감소**



## Overall architecture

![Table 1](Table/1.PNG?raw=true)



## Angular Distillation Loss

- Knowledge distillation은 거대한 네트워크의 능력을 작은 네트워크로 전달하는데 널리 사용됨.

  - 일부 오픈 타스크에서 제한된 정보를 포함하는 훈련 세트 존재
  - 이는 over-regularized 될 수 있음.

- 상기 문제 해결을 위해 angular distillation loss 제안

  - teacher, student의 특징 간 cosine similarity 계산 후 1과 이 유사도 간 L2 distance 최소화

    ![Eq 4](Eq/4.PNG?raw=true)

- 임베딩의 angular information과 분포에 초점을 맞춘 식으로 변경

  ![Eq 5](Eq/5.PNG?raw=true)

- classification loss를 위해 arcface 추가

  ![Eq 6](Eq/6.PNG?raw=true)

- 최종 object function

  - a = 7로 지정

    ![Eq 7](Eq/7.PNG?raw=true)



## Recursive Knowledge Distillation

- knowledge distillation은 때때로 충분한 정보를 전달하기에 어려움이 있음.

  - teacher, student 모델의 어려운 구조 및 블록 셋팅으로 훈련의 복잡도 증가

- **recursive knowledge distillation**을 사용

  - **첫 번째 모델은 그 다음 모델의 사전 학습 모델이 되는 개념**
  - teacher, student 모델 모두 개념 적용
  - 이는, **angular information이 불변**함을 의미

- **이전 세대에서 분류 손실의 한계와 유도 각도 정보 간의 충돌은 다음 세대에서 완화될 것임.**

  ![Fig 2](Fig/2.PNG?raw=true)





## ACC

- TPR(True Positive Rate, 민감도)
  - 1인 케이스에 대해 1로 잘 예측한 비율
  
- FPR(False Positive Rate)
  - 0인 케이스에 대해 1로 잘못 예측한 비율
  
- VarGFaceNet vs y2 (same ephoch)

  ![Table 2](Table/2.PNG?raw=true)

- VarGFaceNet vs Different teacher models

  ![Table 3](Table/3.PNG?raw=true)

- VarGNet vs VarGNetFace

  - Flops : 초당 부동소수점 연산이라는 의미로 컴퓨터가 1초동안 수행할 수 있는 부동소수점 연산의 횟수

  ![Table 4](Table/4.PNG?raw=true)

  