### 간단한 구현을 위해 2D 텐서로 입력받는다 가정(케라스는 3D 텐서임.) ###
import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_dim = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_dim))

hidden_state_t = np.zeros((hidden_size, ))  # 초기 은닉 상태는 0(벡터)로 초기화
"""
>>
    [0. 0. 0. 0. 0. 0. 0. 0.]
"""

Wx = np.random.random((hidden_size, input_dim))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias)

total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs : # 각 시점에 따라서 입력값이 입력됨.
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태의 값을 계속해서 축적
    print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
    
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)     # 출력이 깔끔하게 됨.

print(total_hidden_states) # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.