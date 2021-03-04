# -*- coding: euc-kr -*- 

""" 데이터 전처리 4. Named Entity Recognition using Bi-LSTM과 동일 """
import pandas as pd
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("C:\\Users\\devLupin\\Desktop\\archive\\ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")  # Null 값을 가진 행의 바로 앞의 행의 값으로 Null 값을 채우는 작업 수행
data['Word'] = data['Word'].str.lower()     # 'Word' 행 소문자 화

# 각 문장당 하나의 샘플로 묶는 작업
func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]

# 훈련 데이터에서 단어에 해당되는 부분, 개체명 태깅 정보에 해당되는 부분 분리
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info = zip(*tagged_sentence)
    
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))


""" 토큰화 """
# oov_token은 미리 토큰화 되어 있지 않은 단어를 특정 단어로 처리
# 모든 단어를 사용하지만 인덱스 1에는 단어 'OOV' 할당
src_tokenizer = Tokenizer(oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)
tar_tokenizer = Tokenizer(lower=False)  # 태깅 정보들은 대문자 유지
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1


""" 정수 인코딩 """
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(ner_tags)


word_to_index = src_tokenizer.word_index
index_to_word = src_tokenizer.index_word
ner_to_index = tar_tokenizer.word_index
index_to_ner = tar_tokenizer.index_word
index_to_ner[0] = 'PAD'     # 


""" padding """
max_len = max(len(l) for l in sentences)
avg_len = 70

X_train = pad_sequences(X_train, padding='post', maxlen=avg_len)
y_train = pad_sequences(y_train, padding='post', maxlen=avg_len)


""" data split (train : test = 8 : 2) """
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)


""" one-hot encoding """
y_train = to_categorical(y_train, num_classes=tag_size)
y_test = to_categorical(y_test, num_classes=tag_size)

"""
    Callback 클래스 : F1-score를 측정하는 클래스
        - 모델이 epoch을 도는 순간 모델을 저장하는 것은 불가능
        - 상기 기능을 수행할 수 있는 클래스
"""
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report

class F1score(Callback):
    def __init__(self, value=0.0, use_char=True):
        super(F1score, self).__init__()
        self.value = value
        self.use_char = use_char
        
    # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경
    def sequences_to_tags(self, sequences):
        result = []
        for sequence in sequences:
            tag = []
            
            for pred in sequence:
                pred_index = np.argmax(pred) # 리스트의 값 중 가장 큰 값의 인덱스 반환
                tag.append(index_to_ner[pred_index].replace("PAD", "O"))
                
            result.append(tag)
        return result
    
    # epoch이 끝날 때마다 실행
    def on_epoch_end(self, epoch, logs={}):
        # When use 'char Embedding'
        if self.use_char:
            X_test = self.validation_data[0]
            X_char_test = self.validation_data[1]
            y_test = self.validation_data[2]
            y_predicted = self.model.predict([X_test, X_char_test])
        else:
            X_test = self.validation_data[0]
            y_test = self.validation_data[1]
            y_predicted = self.model.predict([X_test])
            
        pred_tags = self.sequences_to_tags(y_predicted)
        test_tags = self.sequences_to_tags(y_test)
        
        score = f1_score(pred_tags, test_tags)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(test_tags, pred_tags))
        
        if score > self.value:
            print('f1_score improved from %f to %f, saving model to best_model.h5'%(self.value, score))
            self.model.save('best_model.h5')
            self.value = score
        else:
            print('f1_score did not improve from %f'%(self.value))



""" Named Entity Recognition using Bi-LSTM + CRF """
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(50, activation="relu")))    # TimeDistributed는 LSTM이 many-to-many로 동작하게 함.
crf = CRF(tag_size)
model.add(crf)  # 출력층에 crf 층 배치

model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=1, callbacks=[F1score(use_char=False)])

bilstm_crf_model = load_model('best_model.h5', custom_objects={'CRF':CRF, \
                                                                'crf_loss':crf_loss, \
                                                                'crf_viterbi_accuracy':crf_viterbi_accuracy})

i=13 # 확인하고 싶은 테스트용 샘플 인덱스
y_predicted = bilstm_crf_model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경
true = np.argmax(y_test[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))
        


"""
    F1-score 
        - 모델이 리턴하는 예측값은 숫자로 구성되어져 있으므로, 이를 태깅이 나열되어 있는 리스트로 치환해야 함.
            - 이를 위한 함수 sequences_to_tag
"""
f1score = F1score(use_char=False)

y_predicted = bilstm_crf_model.predict([X_test])
pred_tags = f1score.sequences_to_tags(y_predicted)
test_tags = f1score.sequences_to_tags(y_test)

print(classification_report(test_tags, pred_tags))


""" 임의의 문장에 대한 예측 """
new_sentence='Mr. Heo said South Korea has become a worldwide leader'.lower().split()

""" 정수 인코딩 """
new_encoded = []
for w in new_sentence:
    try:
        new_encoded.append(word_to_index.get(w, 1))
    except KeyError:
        new_encoded.append(word_to_index['OOV'])    # # 모델이 모르는 단어에 대해서는 'OOV'의 인덱스인 1로 인코딩
        
print(new_encoded)

""" padding """
new_padded = pad_sequences([new_encoded], padding="post", value=0, maxlen=max_len)

""" predict """
p = bilstm_crf_model.predict(np.array([new_padded[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("단어", "예측값"))
print(30 * "=")     # ==============================과 동일
for w, pred in zip(new_sentence, p[0]):
    print("{:15}: {:5}".format(w, index_to_ner[pred]))