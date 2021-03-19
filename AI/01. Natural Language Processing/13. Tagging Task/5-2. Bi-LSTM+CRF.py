# -*- coding: euc-kr -*- 

""" ������ ��ó�� 4. Named Entity Recognition using Bi-LSTM�� ���� """
import pandas as pd
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("C:\\Users\\devLupin\\Desktop\\archive\\ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")  # Null ���� ���� ���� �ٷ� ���� ���� ������ Null ���� ä��� �۾� ����
data['Word'] = data['Word'].str.lower()     # 'Word' �� �ҹ��� ȭ

# �� ����� �ϳ��� ���÷� ���� �۾�
func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]

# �Ʒ� �����Ϳ��� �ܾ �ش�Ǵ� �κ�, ��ü�� �±� ������ �ش�Ǵ� �κ� �и�
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info = zip(*tagged_sentence)
    
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))


""" ��ūȭ """
# oov_token�� �̸� ��ūȭ �Ǿ� ���� ���� �ܾ Ư�� �ܾ�� ó��
# ��� �ܾ ��������� �ε��� 1���� �ܾ� 'OOV' �Ҵ�
src_tokenizer = Tokenizer(oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)
tar_tokenizer = Tokenizer(lower=False)  # �±� �������� �빮�� ����
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1


""" ���� ���ڵ� """
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
    Callback Ŭ���� : F1-score�� �����ϴ� Ŭ����
        - ���� epoch�� ���� ���� ���� �����ϴ� ���� �Ұ���
        - ��� ����� ������ �� �ִ� Ŭ����
"""
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report

class F1score(Callback):
    def __init__(self, value=0.0, use_char=True):
        super(F1score, self).__init__()
        self.value = value
        self.use_char = use_char
        
    # �������� index_to_ner�� ����Ͽ� �±� ������ ����
    def sequences_to_tags(self, sequences):
        result = []
        for sequence in sequences:
            tag = []
            
            for pred in sequence:
                pred_index = np.argmax(pred) # ����Ʈ�� �� �� ���� ū ���� �ε��� ��ȯ
                tag.append(index_to_ner[pred_index].replace("PAD", "O"))
                
            result.append(tag)
        return result
    
    # epoch�� ���� ������ ����
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
model.add(TimeDistributed(Dense(50, activation="relu")))    # TimeDistributed�� LSTM�� many-to-many�� �����ϰ� ��.
crf = CRF(tag_size)
model.add(crf)  # ������� crf �� ��ġ

model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1, verbose=1, callbacks=[F1score(use_char=False)])

bilstm_crf_model = load_model('best_model.h5', custom_objects={'CRF':CRF, \
                                                                'crf_loss':crf_loss, \
                                                                'crf_viterbi_accuracy':crf_viterbi_accuracy})

i=13 # Ȯ���ϰ� ���� �׽�Ʈ�� ���� �ε���
y_predicted = bilstm_crf_model.predict(np.array([X_test[i]])) # �Է��� �׽�Ʈ�� ���ÿ� ���ؼ� ���� y ����
y_predicted = np.argmax(y_predicted, axis=-1) # ��-�� ���ڵ��� �ٽ� ���� ���ڵ����� ����
true = np.argmax(y_test[i], -1) # ��-�� ���ڵ��� �ٽ� ���� ���ڵ����� ����

print("{:15}|{:5}|{}".format("�ܾ�", "������", "������"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD���� ������.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))
        


"""
    F1-score 
        - ���� �����ϴ� �������� ���ڷ� �����Ǿ��� �����Ƿ�, �̸� �±��� �����Ǿ� �ִ� ����Ʈ�� ġȯ�ؾ� ��.
            - �̸� ���� �Լ� sequences_to_tag
"""
f1score = F1score(use_char=False)

y_predicted = bilstm_crf_model.predict([X_test])
pred_tags = f1score.sequences_to_tags(y_predicted)
test_tags = f1score.sequences_to_tags(y_test)

print(classification_report(test_tags, pred_tags))


""" ������ ���忡 ���� ���� """
new_sentence='Mr. Heo said South Korea has become a worldwide leader'.lower().split()

""" ���� ���ڵ� """
new_encoded = []
for w in new_sentence:
    try:
        new_encoded.append(word_to_index.get(w, 1))
    except KeyError:
        new_encoded.append(word_to_index['OOV'])    # # ���� �𸣴� �ܾ ���ؼ��� 'OOV'�� �ε����� 1�� ���ڵ�
        
print(new_encoded)

""" padding """
new_padded = pad_sequences([new_encoded], padding="post", value=0, maxlen=max_len)

""" predict """
p = bilstm_crf_model.predict(np.array([new_padded[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("�ܾ�", "������"))
print(30 * "=")     # ==============================�� ����
for w, pred in zip(new_sentence, p[0]):
    print("{:15}: {:5}".format(w, index_to_ner[pred]))