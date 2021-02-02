"""
    One-Hot encoding
        - 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값 부여
        - 그렇지 않은 인덱스는 0을 부여
        
        - 과정
            (1) 각 단어에 고유한 인덱스 부여 (Integer encoding)
            (2) 표현하고 싶은 단어의 인덱스 위치에 1 부여, 그 외 0 부여
        
        - 단점
            (1) 단어의 개수가 늘어날 수록 벡터 저장을 위한 공간이 계속 늘어난다.
            (2) 단어의 유사도를 표현할 수 없다.
        - 개선 : 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법을 사용
            (1) 카운트 기반 벡터화 방법인 LSA, HAL 등
            (2) 예측 기반 벡터화 방법인 NNLM, RNNLM, Word2Vec, FastText 등
"""

from konlpy.tag import Okt
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다")   # 형태소 분석기를 통해서 문장에 대한 토큰화 수행

# each token is given index
word2index = {}
for voca in token :
    if voca not in word2index.keys() :
        word2index[voca] = len(word2index)


def one_hot_encoding(word, word2index) :
    one_hot_vector = [0]*(len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

result = one_hot_encoding("자연어", word2index)



### One-Hot Encoding using Keras ###

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

t = Tokenizer()
t.fit_on_texts([text])      # Integer encoding

sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"

"""
    위의 fit_on_texts 처럼 인코딩된 단어 집합이 있다면
    texts_to_sequences()를 통해 정수 시퀸스로 변환 가능
    
    뒤에 [0]은 texts_to_sequences가 list를 반환해서
    반환 형태가 [[list]] 이기 때문임.
"""
encoded=t.texts_to_sequences([sub_text])[0]

one_hot = to_categorical(encoded)   # One-Hot encoding function