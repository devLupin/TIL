# **CRF(Conditional Random Field)**

<hr>

- IO 표현의 제약사항
  - 첫번째 단어의 레이블에서 I가 등장할 수 없음. 
  - I-Per은 반드시 B-Per 뒤에서만 등장
  - I-Org도 마찬가지로 B-Org 뒤에서만 등장
- CRF 층을 추가하면 모델은 예측 개체명(레이블 사이의 의존성) 고려 가능

- **기존에 CRF 층이 존재하지 않았던 양방향 LSTM 모델은 활성화 함수를 지난 시점에서 개체명을 결정**

![img](https://wikidocs.net/images/page/34156/bilstmcrf1.PNG)

- **CRF 층을 추가한 모델에서는 활성화 함수의 결과들이 CRF 층의 입력으로 전달**
  - 모든 단어에 대한 활성화 함수를 지난 출력값은 CRF 층의 입력이 되고, **CRF 층은 레이블 시퀀스에 대해서 가장 높은 점수를 가지는 시퀀스를 예측**

![img](https://wikidocs.net/images/page/34156/bilstmcrf3.PNG)

- CRF 층의 제약사항
  - 문장의 첫 번째 단어에서 I가 나오지 않음.
  - O-I 패턴이 나오지 않음.
  - B-I-I 패턴에서 개체명 일관성 유지
    - ex) B-Per 다음에 I-Org가 나오지 않음.

- **양방향 LSTM은 입력 단어에 대한 양방향 문맥을 반영, CRF는 출력 레이블에 대한 양방향 문맥을 반영**