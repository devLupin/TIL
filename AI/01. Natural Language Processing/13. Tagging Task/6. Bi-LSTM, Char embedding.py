""" 121번 라인부터 """
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




"""
    글자 임베딩을 워드 임베딩과 함께 입력으로 사용
        - 개체명 인식기의 성능을 올리기 위한 방법
        - 워드 임베딩에 글자 임베딩을 연결(concatenate)하여 성능을 높임.
"""

""" 글자 임베딩을 위한 전처리 : 글자 단위 정수 인코딩 """

""" 글자 집합 """
words = list(set(data["Word"].values))
chars = set([w_i for w in words for w_i in w])
chars = sorted(list(chars))

""" 글자 집합 -> 정수, 정수 -> 글자 """
char_to_index = {c: i+2 for i, c in enumerate(chars)}
char_to_index["OOV"] = 1
char_to_index["PAD"] = 0

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key
    
max_len_char = 15

def padding_char_indice(char_indice, max_len_char):
    return pad_sequences(char_indice, maxlen=max_len_char, padding='post', value=0)
    
def integer_coding(sentences):
    char_data = []
    for ts in sentences:
        word_indice = [word_to_index[t] for t in ts]
        char_indice = [[char_to_index[char] for char in t] \
                                            for t in ts]
        
        char_indice = padding_char_indice(char_indice, max_len_char)
        
        for chars_of_token in char_indice:
            if len(chars_of_token) > max_len_char:
                print("Over word length!")
                continue
        
        char_data.append(char_indice)
    return char_data

X_char_data = integer_coding(sentence)  # 글자 단위 정수 인코딩

X_char_data = pad_sequences(X_char_data, maxlen=max_len, padding='post', value = 0)     # 문장 길이 방향 패딩

""" data split """
X_char_train, X_char_test, _, _ = train_test_split(X_char_data, y_data, test_size=.2, random_state=777)
X_char_train = np.array(X_char_train)
X_char_test = np.array(X_char_test)

print(X_train[0])

print(index_to_word[150])
print(' '.join([index_to_char[index] for index in X_char_train[0][0]]))




""" 
    Bi-LSTM and CNN 
        - 글자 단위 정수 인코딩 입력을 1D CNN의 입력으로 사용하여 글자 임베딩을 얻고
            워드 임베딩과 연결하여 양방향 LSTM의 입력으로 사용
"""
from keras.layers import Embedding, TimeDistributed, Dropout, concatenate, Bidirectional, LSTM, Conv1D, Dense, MaxPooling1D, Flatten
from keras import Input, Model
from keras.initializers import RandomUniform
from keras.models import load_model

# 워드 임베딩
word_ids = Input(shape=(None,),dtype='int32',name='words_input')
word_embeddings = Embedding(input_dim = vocab_size, output_dim = 64)(word_ids)

# char 임베딩
char_ids = Input(shape=(None, max_len_char,),name='char_input')
"""
    RandomUniform : 균등 분포에 따라 텐서를 생성하는 초기값 설정기
        minval: 파이썬 스칼라 혹은 스칼라 텐서. 난수값을 생성할 범위의 하한선
        maxval: 파이썬 스칼라 혹은 스칼라 텐서. 난수값을 생성할 범위의 상한선. float 자료형의 경우 디폴트값은 1입니다.
        seed: 파이썬 정수. 난수 생성기에 시드를 전달하는데 사용
"""
embed_char_out = TimeDistributed(Embedding(len(char_to_index), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
dropout = Dropout(0.5)(embed_char_out)

# char 임베딩에 대해서는 Conv1D 수행
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char_embeddings = TimeDistributed(Flatten())(maxpool_out)
char_embeddings = Dropout(0.5)(char_embeddings)

# char 임베딩을 Conv1D 수행한 뒤에 워드 임베딩과 연결
output = concatenate([word_embeddings, char_embeddings])

# 연결한 벡터를 가지고 문장의 길이만큼 LSTM을 수행
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

# 출력층
output = TimeDistributed(Dense(tag_size, activation='softmax'))(output)

model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='nadam',  metrics=['acc'])

history = model.fit([X_train, X_char_train], y_train, \
                    batch_size = 32, epochs = 10, validation_split = 0.1, verbose = 1, callbacks=[F1score(use_char=True)])
# word embedding을 사용하므로 use_char=True

# 모델 로드
bilstm_cnn_model = load_model('best_model.h5')
# 테스트 데이터 F1-score 측정
f1score = F1score(use_char=True)

y_predicted = bilstm_cnn_model.predict([X_test, X_char_test])
pred_tags = f1score.sequences_to_tags(y_predicted)
test_tags = f1score.sequences_to_tags(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))




""" Bi-LSTM and CNN and CRF """
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

# 워드 임베딩
word_ids = Input(shape=(None,),dtype='int32',name='words_input')
word_embeddings = Embedding(input_dim = vocab_size, output_dim = 64)(word_ids)

# char 임베딩
char_ids = Input(shape=(None, max_len_char,),name='char_input')
embed_char_out = TimeDistributed(Embedding(len(char_to_index), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
dropout = Dropout(0.5)(embed_char_out)

# char 임베딩에 대해서는 Conv1D 수행
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char_embeddings = TimeDistributed(Flatten())(maxpool_out)
char_embeddings = Dropout(0.5)(char_embeddings)

# char 임베딩을 Conv1D 수행한 뒤에 워드 임베딩과 연결
output = concatenate([word_embeddings, char_embeddings])

# 연결한 벡터를 가지고 문장의 길이만큼 LSTM을 수행
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

# 출력층에 CRF 층을 추가 (위의 모델과 이 부분이 다름.)
output = TimeDistributed(Dense(50, activation='relu'))(output)
crf = CRF(tag_size)
output = crf(output)

model = Model(inputs=[words_input, character_input], outputs=[output])

model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit([X_train, X_char_train], y_train, \
                    batch_size = 32, epochs = 15, validation_split = 0.1, verbose = 1, callbacks=[F1score(use_char=True)])

bilstm_cnn_crf_model = load_model('best_model.h5', custom_objects={'CRF':CRF,
                                    'crf_loss':crf_loss,
                                    'crf_viterbi_accuracy':crf_viterbi_accuracy})

f1score = F1score(use_char=True)

y_predicted = bilstm_cnn_crf_model.predict([X_test, X_char_test])
pred_tags = f1score.sequences_to_tags(y_predicted)
test_tags = f1score.sequences_to_tags(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))




""" Bi-LSTM and Bi-LSTM and CRF """
# 워드 임베딩
word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
word_embeddings = Embedding(input_dim=vocab_size,
                                        output_dim=64,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)

# char 임베딩
char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
char_embeddings = Embedding(input_dim=(len(char_to_index)),
                                        output_dim=30,
                                        mask_zero=True,
                                        embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                        name='char_embedding')(char_ids)

char_embeddings = TimeDistributed(Bidirectional(LSTM(64)))(char_embeddings)

# char 임베딩을 워드 임베딩과 연결
word_embeddings = concatenate([word_embeddings, char_embeddings])

word_embeddings = Dropout(0.3)(word_embeddings)
z = Bidirectional(LSTM(units=64, return_sequences=True))(word_embeddings)
z = Dense(tag_size, activation='tanh')(z)
crf = CRF(tag_size)
output = crf(z)

model = Model(inputs=[word_ids, char_ids], outputs=[output])
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit([X_train, X_char_train], y_train, \
                    batch_size = 32, epochs = 15, validation_split = 0.1, verbose = 1, callbacks=[F1score(use_char=True)])

bilstm_bilstm_crf_model = load_model('best_model.h5', custom_objects={'CRF':CRF,
                                                'crf_loss':crf_loss,
                                                'crf_viterbi_accuracy':crf_viterbi_accuracy})

f1score = F1score(use_char=True)

y_predicted = bilstm_bilstm_crf_model.predict([X_test, X_char_test])
pred_tags = f1score.sequences_to_tags(y_predicted)
test_tags = f1score.sequences_to_tags(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))