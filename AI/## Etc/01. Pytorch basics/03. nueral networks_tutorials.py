"""
학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망 정의
데이터셋(dataset) 입력 반복
입력을 신경망에서 전파(process)
손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지) 계산
변화도(gradient)를 신경망의 매개변수들에 역으로 전파

신경망의 가중치 갱신
일반적으로 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)
"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

# nn.Module : 매개변수 캡슐화
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input 1, output 6, 3x3의 정사각 conv 행렬
        # conv 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # affine 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
# %%
# parameters : Tensor의 한 종류로, Module에 속성으로 할당될 때 자동으로 매개변수 등록
params = list(net.parameters())
print(len(params))
print(params[0].size())     # conv1의 weight
# %%
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# %%
net.zero_grad()
out.backward(torch.randn(1, 10))
# %%
"""
손실 함수 (Loss Function)
- (output, target)을 한 쌍(pair)의 입력으로 받아, 출력(output)이 정답(target)으로부터 얼마나 멀리 떨어져있는지 추정하는 값 계산

- nn 패키지에는 여러가지의 손실 함수 존재
    - 출력과 대상간의 평균제곱오차(mean-squared error)를 계산하는 nn.MSEloss
"""

output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
# %%
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# %%
net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듦

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# %%
"""
    가중치 갱신
        - 확률적 경사하강법(SGD; Stochastic Gradient Descent)
            - 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)
"""
learning_rate = 0.01
for f in net.parameters():
    # 오차가 역전된 기울기를 바탕으로 파라미터 수정
    f.data.sub_(f.grad.data * learning_rate)
# %%
""" 최적화 """
import torch.optim as optim

# Optimizer 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)
optimizer.zero_grad()   # 기울기 버퍼를 0으로, 기울기 누적 방지
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행
# %%
