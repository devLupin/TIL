# **Data-specific Adaptive Threshold for Face Recognition and Authentication**  

<hr>



## Abstract

- **How to decide the best threshold ?**

## Introduction

- 최근 연구들은 Face verification task에서 threshold를 고정하여 사용
  - 이는 특정 어플리케이션을 제외하면 최적화를 고려하지 않은 셋팅임.
- 임계값을 튜닝해서 경우에 따라 증감하게 만듦.
- **CNN 구조를 사용하여 Embedding feature vector를 추출하고, DB에 해당 threshold와 identity 저장**

## Methodology

- Registration
  - CNN으로부터 입력한 input face에서 feature vector 추출
  - 각 registration 마다 threshold 할당
  - 다른 등록된 얼굴의 threshold도 수정됨.
- Recognition
  - query image 가 주어짐
  - feature embedding을 추출하고, 추출된 것과 다른 저장된 embedding의 유사도 점수 계산
  - 유사도 점수를 사용하여 query image의 identity를 검사

### 1. Deep CNN

- 얼굴의 detect, align을 위해 Multi-task Cascaded Convolutional Networks(MTCNN) 활용
- L2 norm으로 facial feature embedding 추출
  - 두 embedding간의 계산된 유사도를 내적한 값

### 2. Adaptive Threshold

- 각 facial embedding마다 다른 threshold 할당

  - embedding 사이의 유사도 점수 계산

    ![1.PNG](Eq/1.PNG?raw=true)

  - 모든 facial embedding 사이에 최대 유사도 값 계산

    - 같은 것끼리의 비교가 아님.

    ![2.PNG](Eq/2.PNG?raw=true)

  - 이미지는 한번에 하나만 등록됨.

  - t-1, t 시점과 같이 재귀적인 특성으로 효율적인 컴퓨팅 연산 가능

### 3. Recognition and Authentication

- identity label이 없는 query image Iλ가 주어짐.

- CNN으로부터 embedding Fλ 추출

- 모든 embedding에 대한 유사도 점수 계싼

- 가장 높은 유사도 점수 추출

  ![3.PNG](Eq/3.PNG?raw=true)

- 가장 유사한 embedding Fu를 찾고, 등록된 threshold와 유사도 점수 S(λ, u) 와 비교

  - *intruder : 침입자

  ![4.PNG](Eq/4.PNG?raw=true)



## Final ACC

![1.PNG](Table/3.PNG?raw=true)

