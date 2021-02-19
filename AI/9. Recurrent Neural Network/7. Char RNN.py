#%%
import numpy as np
import urllib.request
from tensorflow.keras.utils import to_categorical
# %%
urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")
f = open('11-0.txt', 'rb')
lines=[]
for line in f:
    line = line.strip() # \r, \n 제거
    line = line.lower() # 소문자화.
    line = line.decode('ascii', 'ignore') # \xe2\x80\x99 등과 같은 바이트 열 제거
    
    if len(line) > 0:
        lines.append(line)

f.close()
# %%
lines[:5]
# %%
text = ' '.join(lines)  # 통합
print(len(text))
# %%
print(text[:200])
# %%
char_vocab = sorted(list(set(text)))
vocab_size=len(char_vocab)
# %%
char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
print(char_to_index)
# %%
index_to_char = {}
for key, value in char_to_index.items() :
    index_to_char[value] = key
# %%
seq_length = 60     # 문장의 길이를 60으로 지정
n_samples = int(np.floor((len(text) - 1) / seq_length)) # 문자열 60등분. 즉, 총 샘플의 개수
# %%
train_X = []
train_y = []

for i in range(n_samples) :
    X_sample = text[i * seq_length: (i+1) * seq_length]
    X_encoded = [char_to_index[c] for c in X_sample]    # 하나의 샘플에 대해 정수 인코딩
    train_X.append(X_encoded)
    
    y_sample = text[i * seq_length+1 : (i+1) * seq_length+1]    # shift to right
    y_encoded = [char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)
# %%
print(train_X[0])
print("\n\n")
print(train_y[0])
# %%
"""
    글자 단위 RNN에서는 입력 시퀸스에 대한 워드 임베딩을 진행하지 않음.
    즉, 임베딩 층을 사용하지 않음.
"""
train_X = to_categorical(train_X)
train_y = to_categorical(train_y)
# %%
print('train_X의 크기(shape) : {}'.format(train_X.shape)) # 원-핫 인코딩
print('train_y의 크기(shape) : {}'.format(train_y.shape)) # 원-핫 인코딩
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
# %%
model = Sequential()
model.add(LSTM(256, input_shape=(None, train_X.shape[2]), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=80, verbose=2)
# %%
def sentence_generation(model, length):
    ix = [np.random.randint(vocab_size)] # 글자에 대한 랜덤 인덱스 생성
    y_char = [index_to_char[ix[-1]]]
    print(ix[-1],'번 글자',y_char[-1],'로 예측을 시작!')
    
    X = np.zeros((1, length, vocab_size)) # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성

    for i in range(length):
        X[0][i][ix[-1]] = 1 # X[0][i][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
        print(index_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)  # argmax는 차원에 따라 가장 큰 값의 인덱스를 반환해주는 함수
        y_char.append(index_to_char[ix[-1]])
    return ('').join(y_char)

sentence_generation(model, 100)



""" many-to-one 구조의 RNN 글자 단위 학습 및 텍스트 생성 """
# %%
import numpy as np
from tensorflow.keras.utils import to_categorical

text='''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''
# %%
tokens = text.split()   # '\n' 제거
text = ' '.join(tokens)
print(text)
# %%
char_vocab = sorted(list(set(text)))
print(char_vocab)
# %%
vocab_size = len(char_vocab)
print(vocab_size)   # 33
# %%
char_to_index = dict((c, i) for i, c in enumerate(char_vocab))  # integer index
print(char_to_index)
# %%
length = 11
sequences = []

for i in range(length, len(text)) :
    seq = text[i-length:i]  # 길이 11의 문자열을 지속적으로 만든다.
    sequences.append(seq)

print(len(sequences))   # 426
# %%
sequences[:10]
# %%
sequences[:45]
# %%
X = []
for line in sequences:
    temp_X = [char_to_index[char] for char in line]
    X.append(temp_X)
# %%
""" 분리작업 """
sequences = np.array(X)
X = sequences[:, :-1]
y = sequences[:, -1]
# %%
""" one-hot encoding """
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]  # one-hot encoding for X
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)   # one-hot encoding for 

print(X.shape)
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = Sequential()
model.add(LSTM(80, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)
# %%
# args : 모델, 인덱스 정보, 문장 길이, 초기 시퀸스, 반복 횟수
def sentence_generation(model, char_to_index, seq_length, seed_text, n) :
    init_text = seed_text
    sentence = ''
    
    for _ in range(n) :
        encoded = [char_to_index[char] for char in seed_text]   # integer encoding
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')
        encoded = to_categorical(encoded, num_classes=len(char_to_index))
        result = model.predict_classes(encoded, verbose=0)
        
        for char, index in char_to_index.items() :
            if index == result :    # 예측 글자면?
                break
        
        seed_text = seed_text + char
        sentence = sentence + char
        
    sentence = init_text + sentence
    return sentence
# %%
print(sentence_generation(model, char_to_index, 10, 'I get on w', 80))