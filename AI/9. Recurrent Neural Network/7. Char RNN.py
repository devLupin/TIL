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