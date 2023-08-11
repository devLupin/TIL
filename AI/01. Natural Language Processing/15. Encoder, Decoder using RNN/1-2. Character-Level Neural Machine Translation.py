"""
    기계 번역기 훈련을 위해서는 훈련데이터로 병렬 코퍼스(parallel corpus)가 필요
    병렬 코퍼스 : 두 개 이상의 언어가 병렬적으로 구성된 코퍼스
        - 병렬 데이터는 쌍이 되는 데이터의 길이가 다르다.
    # corpus link
    http://www.manythings.org/anki
"""

import pandas as pd
import urllib3
import zipfile
import shutil   # 파일 복사 함수
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

lines = pd.read_csv('fra_txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
len(lines)

lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000]
print(lines.sample(10))

""" 
시작을 의미하는 심볼 <sos>, 종료를 의미하는 심볼 <eos>가 삽입되어야 함.
<sos> => '\t' <eos> = '\n'으로 간주
"""
lines.tar = lines.tar.apply(lambda x: '\t' + x + '\n')
lines.sample(10)


""" 글자 집합 생성 """
src_vocab = set()
for line in lines.src:
    for char in line:
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1


""" (에러 방지) 정렬하여 순서 지정 """
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

""" 각 글자에 인덱스 부여 """
src_to_index = dict([(word, i+1) for i ,word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i ,word in enumerate(tar_vocab)])

""" 인코더의 입력이 될 영어 문장 샘플에 대한 정수 인코딩 """
encoder_input = []
for line in lines.src:
    temp_X = []
    for w in line:
        temp_X.append(src_to_index[w])
    encoder_input.append(temp_X)

""" 디코더의 입력이 될 프랑스어 데이터에 대한 정수 인코딩 """
decoder_input = []
for line in lines.tar:
    temp_X = []
    for w in line:
        temp_X.append(tar_to_index[w])
    decoder_input.append(temp_X)

""" 실제값은 시작 심볼 <sos>가 필요 없음. 정수 인코딩 과정에서 <sos> 제거 """
decoder_target = []
for line in lines.tar:
    t=0
    temp_X = []
    for w in line:
        if t>0:
            temp_X.append(tar_to_index[w])
        t=t+1
    decoder_target.append(temp_X)

# for padding
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])

""" padding """
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

""" one-hot encoding """
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)



""" 모델 설계 및 교사강요를 사용한 훈련 """
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np

encoder_inputs = Input(shape=(None, src_vocab_size))
# units는 신경망에 존재하는 뉴런의 개수
# return_state=True 인코더의 내부 상태를 디코더로 넘겨줘야 하므로
encoder_lstm = LSTM(units=256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]     # 각각 은닉상태, 셀 상태. 이것이 컨텍스트 벡터이다.

decoder_inputs = Input(shape=(None, tar_vocab_size))
# 디코더의 은닉 상태 크기 또한 256
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
# 디코더는 인코더의 마지막 은닉상태를 초기 은닉상태로 사용. initial_state=encoder_states
# 디코더도 은닉 상태, 셀상태를 리턴하지만 훈련 과정에서는 사용하지 않음.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)
# 출력층에 프랑스어 단어 집합 크기만큼 뉴런을 배치하고 소프트맥스 함수를 사용하여 실제값과의 오차를 구함.
decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=50, validation_split=0.2)



"""
번역 동작 단계
    1. 번역하고자 하는 입력 문장이 인코더에 들어가서 은닉 상태와 셀 상태 확보
    2. 상태와 <SOS>에 해당하는 '\t'를 디코더로 보냄.
    3. 디코더가 <EOS>에 해당하는 '\n'이 나올 때까지 다음 문자를 예측하는 행동 반복
"""

""" 인코더 정의 """
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

""" 디코더 설계 """
# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용
# 이는 뒤의 함수 decode_sequence()에 구현
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
decoder_states = [state_h, state_c]

decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())


# 입력으로부터 인코더의 상태를 얻음
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
    input_seq = encoder_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print('입력 문장:', lines.src[seq_index])
    print('정답 문장:', lines.tar[seq_index][1:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
    print('번역기가 번역한 문장:', decoded_sentence[:len(decoded_sentence)-1]) # '\n'을 빼고 출력