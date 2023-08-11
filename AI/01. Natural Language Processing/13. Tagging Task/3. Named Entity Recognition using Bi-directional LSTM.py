import re
from numpy.lib.shape_base import split
from scipy.sparse.construct import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

f = open('C:\\Users\\devLupin\\Desktop\\TIL\\train.txt', 'r')

tagged_sentences = []
sentence = []

for line in f:
    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":   # 해당 문자로 시작하면 True 반환
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
        
        continue
    
    splits = line.split(' ')
    splits[-1] = re.sub(r'\n', '', splits[-1]) # 줄바꿈 표시 \n을 제거
    word = splits[0].lower()    # 소문자화
    sentence.append([word, splits[-1]])     # 단어, 개체명 태깅만 기록
    
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info = zip(*tagged_sentence)
    
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))
    
max_len = max(len(l) for l in sentences)
max_words = 4000 # 상위 4,000개의 단어만 사용

#oov_token은 미리 토큰화 되어 있지 않은 단어를 특정 단어로 처리
src_tokenizer = Tokenizer(num_words=max_words, oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)

tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size = max_words
tag_size = len(tar_tokenizer.word_index) + 1


""" 정수 인코딩 """
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(ner_tags)


""" 일부 단어가 'OOV'로 대체되었으므로 디코딩 작업 진행 """
index_to_word = src_tokenizer.index_word
index_to_ner = tar_tokenizer.index_word

""" padding """
max_len = 70
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = pad_sequences(y_train, padding='post', maxlen= max_len)

""" train : test = 8 : 2 """
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

""" one-hot encoding """
y_train = to_categorical(y_train, num_classes=tag_size)
y_test = to_categorical(y_test, num_classes=tag_size)



""" Named Entity Recognition using Bi-directional LSTM """
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Many-to-Many 문제이므로 return_sequences=True
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=8, validation_data=(X_test, y_test))

i=10 # 확인하고 싶은 테스트용 샘플의 인덱스
y_predicted = model.predict(np.array([X_test[i]]))  # 입력한 테스트용 샘플에 대해 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1)   # 원-핫 인코딩을 다시 정수 인코딩으로 변경
true = np.argmax(y_test[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t].upper(), index_to_ner[pred].upper()))