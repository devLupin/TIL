from typing import Sequence
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

text="""경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

t = Tokenizer()
t.fit_on_texts([text])

"""
    케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
    케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
    배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언 
"""
vocab_size = len(t.word_index) + 1

sequences = list()
for line in text.split('\n') :
    encoded = t.texts_to_sequences([line])[0]
    
    for i in range(1, len(encoded)) :
        sequence = encoded[:i+1]
        sequences.append(sequence)
        
print(sequences)

max_len = max(len(l) for l in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')     # padding

sequences = np.array(sequences)
X = sequences[:, :-1]   # 리스트의 마지막 값을 제외하고 저장
y = sequences[:, -1]    # 리스트의 마지막 값만 저장. 이는 레이블에 해당

y = to_categorical(y, num_classes=vocab_size)   # 원-핫 인코딩



### RNN model ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)  # verbose=2 는 각 에폭마다 손실정보 출력


"""
    args : model, tokenizer, current word, number of iteration
"""
def sentence_generation(model, t, current_word, n) :
    init_word = current_word
    sentence = ''
    
    for _ in range(n) : # n번 반복
        encoded = t.texts_to_sequences([current_word])[0]   # 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
        result = model.predict_classes(encoded, verbose=0)  # 입력한 현재 단어 X에 대해서 예측한 단어 Y를 result에 저장
        
        for word, index in t.word_index.items() :
            if index == result :
                break
        
        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word
        
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, '경마장에', 4))