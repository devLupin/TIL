# **Face Detection using Haar Cascades**

<hr>
[1] : Paul Viola, Michael Jones,. "Rapid Object Detection using a Boosted Cascade of Simple Features", in *CVPR 2001*



## Summary

- 검출할 대상이 되는 물체가 있는 이미지와 없는 이미지(각각을 Positive Image와 Negative Image라고 함)을 최대한 많이 활용해서 다단계 함수를 훈련시키는 기계학습 방식

- 각 특징은 하얀색 사각형에서의 픽셀값의 합을 검정색 사각형 영역의 픽셀값의 합에서 뺀 값

  - Convolutional kernel과 유사함.

    ![img](http://www.gisdeveloper.co.kr/wp-content/uploads/2019/06/HaarFeatures.png)

- 많은 특징을 계산하기 위해 이미지에 적용할 각 커널에 대한 모든 가능한 크기와 위치를 고려해야 함.



## **Integral Image**

- 다음 픽셀에 이전 픽셀까지의 합이 더해진 형태

- 하나의 이미지에 대해 특정 영역의 픽셀 값의 합을 여러 번 구해야 할 필요가 있을 때 유용하게 사용

  - 이미지 내부의 특정 영역의 내부합을 구할때 간단한 연산으로 구할 수 있음.

- (0,0) 부터 시작해서 (n, m) 까지 픽셀성분값의 누적값이 Integral Image의 픽셀값

- 예시

  - D 영역의 픽셀 값을 얻기 위해서 점 d까지의 넓이에서 점 b까지의 넓이와 점 c까지의 넓이를 뺀 후

    두 번 빼진 점 a 까지의 넓이를 한 번 더해줌으로써 D 영역의 넓이를 구할 수 있음.

    ![img](https://t1.daumcdn.net/cfile/tistory/171A2648505508D421)

- 논문에서 언급된 24-by-24 크기의 이미지를 계산해도 엄청난 양의 연산 필요

  - **Integral image**를 사용하여 알고리즘 속도를 매우 빠르게 함.

    - 이렇게 계산된 특징 대부분이 적합하지 않다고 함.

      - 특정 영역의 계산된 특징 외에는 모두 쓸모 없음.

        ![img](http://www.gisdeveloper.co.kr/wp-content/uploads/2019/06/haar.png)

    - 수많은 특징 중 최고의 특징만을 선택해야 하는 문제 직면



## **Adaboost**

- 언급된 문제를 **Adaboost**를 이용하여 해결
  - 모든 훈련 이미지들에 대해 각각의 모든 특징 적용
  - 각 특징에 대해, 이미지 상에 얼굴이 있는지 없는지(즉 positive 또는 negative 인지)를 분류할 최고의 임계값 탐색
    - 이는 명백한 오류와 잘못된 분류 존재
  - 수행 절차
    1. 각 이미지는 동일한 가중치가 주어짐.
    2. 각각의 분류 후, 잘못 분류된 이미지의 가중치 증가시킴.
    3. 1~2 반복
       - 새로운 에러율이 얻어지거나 원하는 개수의 특징점 발견
  - 최종 classifier는 이러한 약한 classifier들의 가중치 합
    - 논문에서는 200개의 특징만으로도 90%의 정확도를 제공한다고 언급
  - 160,000개의 특징을 6,000개의 특징점으로 감소시킴.



## **Cascade of Classifier**

- 각 이미지에 24-by-24의 윈도우를 지정하고 6,000개의 특징을 이미지에 적용하여 얼굴 유무 확인
  - 매우 높은 시간적 비용 발생
- **얼굴이 아닌 영역(배경 등)이라면, 연산을 안하고 바로 제거, 다시는 수행하지 않는 개념 도입**
- 수행절차
  1. 6,000개의 모든 특징을 적용하지만, classifier의 다른 단계로 특징을 귀속시키고 하나씩 적용
     - 처음 몇 단계는 매우 적은 수의 특징을 지니게 됨.
  2. 윈도우가 첫 단계에서 실패하면 버리고, 나머지 특징을 고려하지 않음.
  3. 만약 통과되면 두번째 단계를 적용
  4. 1~3 반복
     - 모든 단계를 통과한 윈도우가 얼굴 영역이 됨.



## Usage

- OpenCV는 XML 파일 형식으로 pre-trained 데이터 제공

- 예시

  ```python
  import numpy as np
  import cv2
  from matplotlib import pyplot as plt
  
  face_cascade = cv2.CascadeClassifier('./data/haar/haarcascade_frontface.xml') 	# face classifier XML load
  eye_cascade = cv2.CascadeClassifier('./data/haar/haarcascade_eye.xml')		# eye classifier XML load
  img = cv2.imread('./data/haar/img.jpg') 	# Input the sample
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 	# Convert to Grayscale
  
  """
  detectMultiScale
  	- 입력 이미지에서 크기가 다른 물체 감지
  	- params:
  		image 			물체가 감지 된 이미지를 포함하는 CV_8U 유형의 매트릭스
  		scaleFactor		각 이미지 축척에서 이미지 크기가 얼마나 줄어드는지 지정
  		minNeighbors 	각 후보 사각형이 유지해야하는 이웃 수 지정
  	- return:
  		Rect(x,y,w,h) 	list of rectangles (If find face)
  """
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
  for (x,y,w,h) in faces:
      """
      rectangle
      	- 사각형을 그리는 함수
      	- params:
      		img			입력 이미지
      		start		시작 좌표
      		end			종료 좌표
      		color		RGB 값
      		thickness	선의 두께
      """
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
      
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray) 	# 눈 영역 검출
      for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  
  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

- 결과 예

  ![img](http://www.gisdeveloper.co.kr/wp-content/uploads/2019/06/face_detecting.png)



## Discussions

- 훈련 데이터가 정면에서 본 얼굴에 대한 이미지로 만들어져 있으므로, 그렇지 못한 얼굴은 검출되지 못함.
  - Face alignment 기술 필요