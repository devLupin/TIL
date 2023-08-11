# **Masked Face Recognition[1]**

<hr>
[1] : "Masked Face Recognition Dataset and Application", in *CVPR*



## Proposed Datasets

- Face mask detection, Masked recognition task로 나뉨.

- Mask detection task

  - 경우에 따라 마스크를 썼는지 확인 필요

- Masked recognition task

  - 마스크를 착용한 사람 식별

- Masked Face Detection Dataset (**MFDD**), Real-world Masked Face Recognition Dataset (**RMFRD**), Simulated Masked Face Recognition Dataset (**SMFRD**)으로 구성

  - MFDD

    - 인터넷 크롤링된 데이터
    - 마스크를 착용한 얼굴,  마스크된 얼굴의 좌표 정보로 구성
    - Masked recognition task의 정확도 향상을 위해 사용

  - RMFRD

    - 전면 얼굴 이미지와 그에 해당하는 마스크된 얼굴 이미지 크롤링

    - 잘못 해당된 비합리적 얼굴 이미지는 제거(수작업으로 진행)

    - image crop(얼굴 영역만을 자름.)을 위해 LabelImg, LabelMe tool 사용

      - https://tzutalin.github.io/labelImg/

      ![Fig 1](Fig/1.PNG?raw=true)

  - SMFRD

    - 이미 존재하는 데이터셋 LFW, Webface와 같은 데이터셋을 이용하여 마스크가 착용된 듯한 이미지 생성

    - mask wearing software based on Dlib library를 사용하여 자동화

      - http://dlib.net/

      ![Fig 2](Fig/2.PNG?raw=true)



## Masked Face Recognition

- 제어된 어플리케이션 시나리오(직장에서의 출석, 보안 체크, 얼굴 스캔 결제 등)의 시나리오
  - Cooperative manner(협력적 방식) : 일반적으로, 카메라 전면을 향하게 함.
- **마스크 간섭을 배제하고 유용한 노출 얼굴 특징에 더 높은 우선 순위 부여**
- Two aspect
  - Bulit dataset
    - 상기 3가지 데이터셋 구성 참조
  - Uncovered useful face feature 사용
    - 마스크를 착용한 얼굴에서 보여지는 부분의 특징(얼굴윤곽, 안구, 눈 주위, 이마)등의 특징을 주요 가중치로 부여
    - 얼굴 식별 정보의 불균등한 분포 문제를 해결
- ACC
  - 초기 상태에서는 50%의 정확도를 얻었고, 데이터를 좀 더 수집하여 95%의 정확도를 달성했다고 언급됨.