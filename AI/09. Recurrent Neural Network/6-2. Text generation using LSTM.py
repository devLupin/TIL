#%%
import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('..\#. Dataset\ArticlesApril2018.csv')
df.head()
# %%
print(df.columns)
# %%
df['headline'].isnull().values.any()
# %%
headline = []
headline.extend(list(df.headline.values))   # headline 값들 저장
headline[:5]
# %%
print(len(headline))
# %%
headline = [n for n in headline if n != "Unknown"]
print(len(headline))
# %%
### 구두점 제거 및 소문자화 ###
def repreprocessing(s) :
    s = s.encode("utf8").decode("ascii", 'ignore')
    return ''.join(c for c in s if c not in punctuation).lower()

text = [repreprocessing(x) for x in headline]
text[:5]
# %%
t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1
print(vocab_size)
# %%
sequences = list()  # train set(X)

for line in text :
    encoded = t.texts_to_sequences([line])[0]
    
    for i in range(1, len(encoded)) :
        sequence = encoded[:i+1]
        sequences.append(sequence)

sequences[:11]
# %%
index_to_word = {}
for key, value in t.word_index.items() :
    index_to_word[value] = key
# %%
### padding ###
max_len = max(len(l) for l in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
# %%
sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
# %%
y = to_categorical(y, num_classes=vocab_size)   # 정수 인코딩
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
# %%
model = Sequential()
""" y데이터를 분리했으므로 x데이터의 길이는 기존 데이터의 길이 -1 """
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
# %%
def sentence_generation(model, t, current_word, n) :
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=23, padding='pre')
        result = model.predict_classes(encoded, verbose=0)
        
        for word, index in t.word_index.items() :
            if index == result :
                break
            current_word = current_word + ' ' + word
            sentence = sentence + ' ' + word
            
    sentence = init_word + sentence
    return sentence
# %%
print(sentence_generation(model, t, 'i', 10))
# %%
print(sentence_generation(model, t, 'how', 10))