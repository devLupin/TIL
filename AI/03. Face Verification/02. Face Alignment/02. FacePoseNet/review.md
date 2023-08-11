# **FacePoseNet**

<hr>

[1] : "FacePoseNet: Making a Case for Landmark-Free Face Alignment" in *CVPR 2017*



## Abstract

- landmark detector 정확도는 서로 다른 얼굴 정렬을 비교할 때 문제 존재



## Introduction

- 일반적으로 5, 49, 68개의 랜드마크 사용
- 넓은 뷰포인트, 해상도, 노이즈 반영
- 불확실한 사람 label의 더 나은 추정을 위해 제안



## Related work

### 1. Align a face

- facial landmark detection method
- 특정 공간 위치 간 대응 형성
- 더 쉬운 비교 및 매치



## A critique of facial landmark detection  

### 1. Landmark detection accuracy measures

- 일반적으로 추정된 랜드마크와 GT 간 거리 고려로부터 측정

- +얼굴의 기준 안구 간 거리로 정규화

- 특정 오류 임계값 또는 AUC 아래 영역에서 감지된 랜드마크의 퍼센트

- Fomular

  - pi : 2D facial landmark coordinates

  - pi hat : GT locations

  - pl, pr : left and right eye outer corner positions

    ![Eq 1](Eq/1.PNG?raw=true)

- 문제점

  - 때때로 사람이 수동으로 지정
  - 오류 측정 자체에서 문제 발생
    - 비정면인 얼굴 이미지에 대한 안구 간 거리 차로 탐지 오류 정규화
    - 3D 얼굴 원근 투영은 눈 사이의 거리를 축소하여 계산된 오류를 부풀림.

### 2. Landmark detection speed

- 최신 연구 중 랜드마크 감지기가 없는 모델 다수 존재
  - 병렬 처리가 어려울 수 있음.

### 3. Effects on alignment

- 3D 정면화, 얼굴의 뒤틀림은 3D, 2D 관계없이 효과적
- 최근 렌더링 기술과 더불어, 일반적인 3D face shape은 2D image와 같은 속도로 정면화 가능
- 파라메트릭 변환 계산을 위해 얼굴 랜드마크 감지기 사용