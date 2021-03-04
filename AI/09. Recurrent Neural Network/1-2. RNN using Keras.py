from keras.models import Sequential
from keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2,10)))     # == model.add(SimpleRNN(3, input_length=2, input_dim=10))
model.summary()
"""
현재 배치사이즈를 입력하지 않아 None으로 표기됨.
>>
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    simple_rnn (SimpleRNN)       (None, 3)                 42
    =================================================================
    Total params: 42
    Trainable params: 42
    Non-trainable params: 0
    _________________________________________________________________
"""



### batch_size 미리 정의 ###
model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10)))
model.summary()
"""
>>
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    simple_rnn_1 (SimpleRNN)     (8, 3)                    42
    =================================================================
    Total params: 42
    Trainable params: 42
    Non-trainable params: 0
    _________________________________________________________________
"""



### (batch_size, timesteps, output_dim) 크기의 3D 텐서를 리턴 ###
model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))
model.summary()
"""
>>
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn_3 (SimpleRNN)    (8, 2, 3)                 42        
    =================================================================
    Total params: 42
    Trainable params: 42
    Non-trainable params: 0
    _________________________________________________________________
"""