""" 121�� ���κ��� """
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




"""
    ���� �Ӻ����� ���� �Ӻ����� �Բ� �Է����� ���
        - ��ü�� �νı��� ������ �ø��� ���� ���
        - ���� �Ӻ����� ���� �Ӻ����� ����(concatenate)�Ͽ� ������ ����.
"""

""" ���� �Ӻ����� ���� ��ó�� : ���� ���� ���� ���ڵ� """

""" ���� ���� """
words = list(set(data["Word"].values))
chars = set([w_i for w in words for w_i in w])
chars = sorted(list(chars))

""" ���� ���� -> ����, ���� -> ���� """
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

X_char_data = integer_coding(sentence)  # ���� ���� ���� ���ڵ�

X_char_data = pad_sequences(X_char_data, maxlen=max_len, padding='post', value = 0)     # ���� ���� ���� �е�

""" data split """
X_char_train, X_char_test, _, _ = train_test_split(X_char_data, y_data, test_size=.2, random_state=777)
X_char_train = np.array(X_char_train)
X_char_test = np.array(X_char_test)

print(X_train[0])

print(index_to_word[150])
print(' '.join([index_to_char[index] for index in X_char_train[0][0]]))




""" 
    Bi-LSTM and CNN 
        - ���� ���� ���� ���ڵ� �Է��� 1D CNN�� �Է����� ����Ͽ� ���� �Ӻ����� ���
            ���� �Ӻ����� �����Ͽ� ����� LSTM�� �Է����� ���
"""
from keras.layers import Embedding, TimeDistributed, Dropout, concatenate, Bidirectional, LSTM, Conv1D, Dense, MaxPooling1D, Flatten
from keras import Input, Model
from keras.initializers import RandomUniform
from keras.models import load_model

# ���� �Ӻ���
word_ids = Input(shape=(None,),dtype='int32',name='words_input')
word_embeddings = Embedding(input_dim = vocab_size, output_dim = 64)(word_ids)

# char �Ӻ���
char_ids = Input(shape=(None, max_len_char,),name='char_input')
"""
    RandomUniform : �յ� ������ ���� �ټ��� �����ϴ� �ʱⰪ ������
        minval: ���̽� ��Į�� Ȥ�� ��Į�� �ټ�. �������� ������ ������ ���Ѽ�
        maxval: ���̽� ��Į�� Ȥ�� ��Į�� �ټ�. �������� ������ ������ ���Ѽ�. float �ڷ����� ��� ����Ʈ���� 1�Դϴ�.
        seed: ���̽� ����. ���� �����⿡ �õ带 �����ϴµ� ���
"""
embed_char_out = TimeDistributed(Embedding(len(char_to_index), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
dropout = Dropout(0.5)(embed_char_out)

# char �Ӻ����� ���ؼ��� Conv1D ����
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char_embeddings = TimeDistributed(Flatten())(maxpool_out)
char_embeddings = Dropout(0.5)(char_embeddings)

# char �Ӻ����� Conv1D ������ �ڿ� ���� �Ӻ����� ����
output = concatenate([word_embeddings, char_embeddings])

# ������ ���͸� ������ ������ ���̸�ŭ LSTM�� ����
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

# �����
output = TimeDistributed(Dense(tag_size, activation='softmax'))(output)

model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='nadam',  metrics=['acc'])

history = model.fit([X_train, X_char_train], y_train, \
                    batch_size = 32, epochs = 10, validation_split = 0.1, verbose = 1, callbacks=[F1score(use_char=True)])
# word embedding�� ����ϹǷ� use_char=True

# �� �ε�
bilstm_cnn_model = load_model('best_model.h5')
# �׽�Ʈ ������ F1-score ����
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

# ���� �Ӻ���
word_ids = Input(shape=(None,),dtype='int32',name='words_input')
word_embeddings = Embedding(input_dim = vocab_size, output_dim = 64)(word_ids)

# char �Ӻ���
char_ids = Input(shape=(None, max_len_char,),name='char_input')
embed_char_out = TimeDistributed(Embedding(len(char_to_index), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
dropout = Dropout(0.5)(embed_char_out)

# char �Ӻ����� ���ؼ��� Conv1D ����
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char_embeddings = TimeDistributed(Flatten())(maxpool_out)
char_embeddings = Dropout(0.5)(char_embeddings)

# char �Ӻ����� Conv1D ������ �ڿ� ���� �Ӻ����� ����
output = concatenate([word_embeddings, char_embeddings])

# ������ ���͸� ������ ������ ���̸�ŭ LSTM�� ����
output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

# ������� CRF ���� �߰� (���� �𵨰� �� �κ��� �ٸ�.)
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
# ���� �Ӻ���
word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
word_embeddings = Embedding(input_dim=vocab_size,
                                        output_dim=64,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)

# char �Ӻ���
char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
char_embeddings = Embedding(input_dim=(len(char_to_index)),
                                        output_dim=30,
                                        mask_zero=True,
                                        embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                        name='char_embedding')(char_ids)

char_embeddings = TimeDistributed(Bidirectional(LSTM(64)))(char_embeddings)

# char �Ӻ����� ���� �Ӻ����� ����
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