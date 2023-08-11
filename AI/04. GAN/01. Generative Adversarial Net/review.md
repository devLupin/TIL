# **Generative Adversarial Nets**

<hr>

- Deep generative model은 다루기 힘든 확률 계산을 근사하기 어려움
  - 최대 가능성 추정, 관련 전략에서 발생하는 확률 계산
  - 생성적 맥락에서 선형 단위의 이점 활용이 어려움
- 이러한 어려움을 피하는 새로운 생성 모델의 제안

- 제안된 adversarial net 프레임워크에서 generative 모델은 discriminative 모델과 싸우게 됨.
  - discriminative model
    - 샘플이 모델 분포에서 나온 것인지 아니면 데이터 분포에서 나온 것인지 검증
    - 경찰과 유사하며, 위조 화폐를 탐지하려고 함.
  - generative model
    - 가짜 화폐를 만들어서 탐지 없이 사용하려는 위조자 팀이라고 생각할 수 있음.
  - 경쟁을 통해 위조품과 정품을 구별할 수 없을 때까지 discriminative, generative의 기능을 향상 시킴.
  - 두 모델은 Multi-layer perceptron으로 구성됨.

- adversarial nets

  - generative, discriminative model이 다층 퍼셉트론을 통해 랜덤 노이즈를 전달하여 샘플을 생성하는 경우

  - 두 모델은 높은 성공률의 역전파와 드롭아웃 알고리즘만을 이용하여 학습 시킴.

  - 샘플은 순전파를 이용한 생성 모델로부터 학습시킴.

  - **샘플 데이터의 분포를 학습한다.**

  - Generative(G), Discriminative(D)

    ![img](https://blog.kakaocdn.net/dn/cocQG4/btqEPllaEt6/La3SQp6ksuWWcTnOLypKRk/img.png)



## Result

<hr>

- 처음엔 생성된 데이터와 샘플 데이터의 분포가 큰 차이가 나지만 점차 분포가 유사해지고, 진품과 가품을 구별하지 못하는 이상적 상태가 된다.

- blue - D, black - sample, green - G

  ![img](https://blog.kakaocdn.net/dn/cYLxST/btqERsJITK3/Vruhky37COwYyFk7nKLaF1/img.png)

- SGD를 사용하여 D 업데이트 후, G 업데이트

  ![img](https://blog.kakaocdn.net/dn/bHxruV/btqEPlZKJ59/eUpChhDAQlSzeEWUUafDY1/img.png)

- 최적화

  - G를 고정시킨 후, 최적의 판별기 D

    - G를 통해 만든 확률 분포와 data를 통해 만든 확률 분포가 같을 때 최적

    ![img](https://blog.kakaocdn.net/dn/2o8cM/btqEP3c4xEs/F7shUXkKNM9ps2skTTkQq0/img.png)



## Advantages and disadvantages

<hr>

- Markov chains이 필요없고 역전파만 필요
- G는 data로 부터 직접 업데이트되지 않고 D로 업데이트
-  이전의 Markov모델들 보다 sharp한 특징을 잡아낼 수 있음.

- 학습하는 동안 G와 D가 합이 잘 맞아야 함.