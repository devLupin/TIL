""" 전처리(Preprocessing) """

from tensorflow.keras.preprocessing.text import Tokenizer
t = Tokenizer()     # 토큰화와 정수 인코딩(단어에 대한 인덱싱)을 위해 사용
fit_next = "The earth is an awesome place live"
t.fit_on_texts([fit_next])

test_text = "The earth is an great place live"
sequences = t.texts_to_sequences([test_text])[0]

print("sequences : ",sequences) # great는 단어 집합(vocabulary)에 없으므로 출력되지 않는다.
print("word_index : ",t.word_index) # 단어 집합(vocabulary) 출력
"""
>>
    sequences :  [1, 2, 3, 4, 6, 7]
    word_index :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
"""



""" 패딩(padding) """
"""
    각 문서 또는 각 문장의 단어의 수를 동일하게 맞춰주는 작업
    보통 숫자 0을 넣어서 길이를 맞춤.
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')
"""
    padding=''
        pre(앞에서부터), post(뒤에서부터)
>>
    array([[1, 2, 3],
           [4, 5, 6],
           [0, 7, 8]], dtype=int32)
"""



"""
    워드 임베딩(Word Embedding)

        텍스트 내의 단어들을 밀집 벡터(dense vector)로 만드는 것
        상대적으로 저차원이며 실수값을 가짐.

        임베딩 벡터는 주로 256, 512, 1024 등의 차원을 가짐.
        임베딩 벡터는 초기에는 랜덤값을 가지지만, 가중치가 학습되는 방법과 같은 방식으로 값이 학습되며 변경됨.

        원-핫 벡터 : [0 1 0 0 0 0]
        밀집 벡터 : [0.1 -1.2 0.8 0.2 1.8]
"""

# 문장 토큰화와 단어 토큰화
text=[['Hope', 'to', 'see', 'you', 'soon'],['Nice', 'to', 'see', 'you', 'again']]

# 각 단어에 대한 정수 인코딩
text=[[0, 1, 2, 3, 4],[5, 1, 2, 3, 6]]

# 위 데이터가 아래의 임베딩 층의 입력이 된다.
Embedding(7, 2, input_length=5)
# 7은 단어의 개수. 즉, 단어 집합(vocabulary)의 크기이다.
# 2는 임베딩한 후의 벡터의 크기이다.
# 5는 각 입력 시퀀스의 길이. 즉, input_length이다.
"""
    Embedding은 2D 텐서를 입력받음.
    이 때 각 샘플은 정수 인코딩이 된 결과로 정수의 시퀸스
    3D 텐서 리턴
"""



""" 모델링(Modeling) """
from tensorflow.keras.models import Sequential
model = Sequential()    # 층 구성
model.add(...)  # 층을 단계적으로 추가
# 임베딩 층도 add()를 통해 추가해야 함.
model.add(Dense(1, input_dim=3, activation='relu'))     # Fully-connected layer 추가
"""
    activation='활성화 함수'
        - linear : 디폴트 값으로 별도 활성화 함수 없이 입력 뉴런과 가중치의 계산 결과 그대로 출력. Ex) 선형 회귀
        - sigmoid : 시그모이드 함수. 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
        - softmax : 소프트맥스 함수. 셋 이상을 분류하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
        - relu : 렐루 함수. 은닉층에 주로 사용되는 활성화 함수.
"""

model.summary()     # 모델의 정보를 요약해서 보여줌.

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])     # 모델을 기계가 이해할 수 있도록
"""
    optimizer : 훈련 과정 설정
    loss : 훈련과정에서 사용할 손실 함수 설정
    metrics : 훈련을 모니터링하기 위한 지표
    
    문제 유형	                손실 함수명	                                출력층의 활성화 함수명	        참고 설명
    회귀 문제	        mean_squared_error(평균 제곱 오차)	                            -	                        -
    다중 클래스 분류	categorical_crossentropy (범주형 교차 엔트로피)	            소프트맥스	            
    다중 클래스 분류	sparse_categorical_crossentropy	                            소프트맥스	          범주형 교차 엔트로피와 동일하지만 이 경우 원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태에서 수행 가능.
    이진 분류	        binary_crossentropy(이항 교차 엔트로피)	                    시그모이드	            
"""

# 모델 학습. 모델이 오차로부터 매개변수를 업데이터 시키는 과정
model.fit(train_data, lable_data, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val))
"""
    validation_data(x_val, y_val)
        - 검증 데이터(validation data) 사용 
        - 검증 데이터를 사용하면 각 에포크마다 검증 데이터의 정확도도 함께 출력되는데, 이 정확도는 훈련이 잘 되고 있는지를 보여줄 뿐
        - 검증 데이터의 loss가 낮아지다가 높아지기 시작하면 이는 과적합(overfitting)의 신호
        
    validation_split
        - validation_data 대신 사용 가능
        - 검증 데이터를 사용하는 것은 동일하지만, 별도로 존재하는 검증 데이터를 주는 것이 아니라 X_train과 y_train에서 일정 비율을 분리하여 이를 검증 데이터로 사용
        
    verbose
        - 학습 중 출력되는 문구 설정
        - 0 : 아무 것도 출력하지 않습니다.
        - 1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다.
        - 2 : 미니 배치마다 손실 정보를 출력합니다.
"""

model.evaluate(test_data, lable_data, batch_size=32)   # 모델에 대한 정확도 평가

model.predict(predict_input, batch_size=32)     # 임의의 입력에 대한 모델의 출력값 확인

model.save("model_name.h5")     # 모델 hdf5 파일에 저장
from tensorflow.keras.models import load_model
model = load_model("model_name.h5")     # 저장해둔 모델 로드