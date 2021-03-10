import nltk
from nltk.tag import pos_tag
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.variables import trainable_variables

tagged_sentences = nltk.corpus.treebank.tagged_sents() # 토큰화에 품사 태깅이 된 데이터 받아오기


""" 훈련을 위한 단어, 품사 태깅 정보 분리 """
sentences, pos_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info = zip(*tagged_sentence)
    
    sentences.append(list(sentence))
    pos_tags.append(list(tag_info))

def tokenize(samples):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(samples)
    return tokenizer

src_tokenizer = tokenize(sentences)     # 문장 데이터
tar_tokenizer = tokenize(pos_tags)      # 품사 태깅 정보

vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1


""" 정수 인코딩 """
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(pos_tags)


""" padding """
max_len = max(len(l) for l in sentences)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

# train : test = 8 : 2
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)


""" one-hot encoding """
y_train = to_categorical(y_train, num_classes=tag_size)
y_test = to_categorical(y_test, num_classes=tag_size)



""" POS Tagger to Bi-directional LSTM """
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from keras.optimizers import Adam

model = Sequential()
# mask_zero=True는 0으로 패딩된 값을 마스킹하여 네트워크의 뒤로 전달되지 않게 만든다.
# 이렇게 하면 인위적으로 패딩된 부분은 학습에 영향을 미치지 않는다.
model.add(Embedding(vocab_size, 128, input_length=max_len, mask_zero=True))
# return_sequences 인자에 따라 마지막 시퀀스에서 한 번만 출력, 각 시퀀스에서 출력 가능
# many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 return_sequence=True 옵션 사용
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=6, validation_data=(X_test, y_test))

index_to_word=src_tokenizer.index_word
index_to_tag=tar_tokenizer.index_word

i=10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_tag[t].upper(), index_to_tag[pred].upper()))