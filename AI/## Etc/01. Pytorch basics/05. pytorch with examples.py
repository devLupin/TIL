"""
    PyTorch
        - GPU 상에서 실행 가능한 N차원 Tensor
        - 신경망을 구성하고 학습하는 과정에서의 자동 미분
"""
#%%
""" Numpy """
import numpy as np

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 입출력 데이터 랜덤 생성
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 랜덤 가중치 초기화
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # forward step : calc predict y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # calc & print loss
    # square == ** 2
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    # (according loss) calc w1, w2 of gradient and backward
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    # update weight
    # SGD - 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
# %%
""" PyTorch """
import torch

dtype = torch.float
device = torch.device("cuda:0")

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터 생성
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 무작위로 가중치 초기화
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 순전파 : 예측값 y 계산
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # loss 계산, 출력
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 w1, w2의 기울기 계산 후 역전파
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 경사하강법(gradient descent)를 사용하여 가중치 갱신
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
# %%
""" PyTorch, autograd """
import torch

dtype = torch.float
device = torch.device("cuda:0")

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

# 무작위의 입력과 출력 데이터 생성
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 무작위로 가중치 초기화
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 순전파 : 예측값 y 계산
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss 계산, 출력
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 손실에 따른 w1, w2의 기울기 계산 후 역전파
    """
    - autograd를 사용하여 역전파 계산
    - requires_grad=True를 갖는 모든 Tensor에 대한 loss의 기울기 계산
    - var.grad는 각각의 손실 기울기를 갖는 Tensor가 됨.
    """
    loss.backward()

    # 경사하강법(gradient descent)를 사용하여 가중치 수동 갱신
    """
    - torch.no_grad()로 감싸는 이유는 가중치들이 requires_grad=True 이지만
    autograd에서는 이를 추적할 필요가 없기 때문임.
    - torch.optim.SGD를 사용해도 됨.
    """
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로
        w1.grad.zero_()
        w2.grad.zero_()
# %%
""" User defined autograd """
"""
    - PyTorch autograd의 기본 연산자는 Tensor를 조작하는 forward, backward로 구성
    - forward : 입력 Tensor로부터 출력 Tensor 계산
    - backward : 어떤 스칼라 값에 대한 출력 Tensor의 변화도를 전달받고, 동일한 스칼라 값에 대한 입력 Tensor의 변화도 계산

    - torch.autograd.Function의 subclass를 정의하고 forward, backward 함수를 구현함으로써
    사용자 정의 autograd 연산자를 정의할 수 있음.
    - 이후, instance를 생성하고, 이를 함수처럼 호출하여 입력 데이터를 갖는 Tensor를 전달하는 형태
"""
import torch

class MyReLU(torch.autograd.Function):
    """
    - torch.autograd.Function을 상속받아 사용자 정의 autograd Function을 구현
    - Tensor 연산을 하는 순전파와 역전파 단계를 구현
    """
    
    @staticmethod
    def forward(ctx, input):
        """
            - 입력을 갖는 Tensor를 받아 출력을 갖는 Tensor 반환
            - ctx는 컨텍스트 객체(context object)로 역전파 연산을 위한 정보 저장
            - ctx.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한 객체도 저장(cache) 가능
        """
        
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
            - 출력에 대한 손실의 변화도를 갖는 Tensor 입력
            - 입력에 대한 손실의 변화도 계산
        """
        
        input, = ctx.saved_tensors
        # clone() : 텐서 복사
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cuda:0")

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    relu = MyReLU.apply # 사용자 정의 Function 적용
    
    # forward : Tensor 연산을 사용하여 예상되는 y 값 계산
    # 사용자 정의 autograd 연산을 사용하여 ReLU 계산
    y_pred = relu(x.mm(w1)).mm(w2)
    
    # calc loss
    loss = (y_pred -y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    
    # backward using autograde
    loss.backward()
    
    # update weight using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        w1.grad.zero_()
        w2.grad.zero_()
# %%
""" PyTorch: nn """
"""
    - 신경망 구성 시, 연산을 여러 계층에 배열(arrange)하는데, 이 중 일부는 학습 도중 최적화 될
    학습 가능한 매개변수를 가짐.
    - Tensorflow의 Keras와 동일한 목적으로 제공되는 패키지
    - nn 패키지는 신경망 layer와 거의 동일한 Moudle의 집합 정의
        -  Module은 입력 Tensor를 받고 출력 Tensor 계산하는 한편, 학습 가능한 매개변수를 갖는 Tensor 같은 내부 상태(internal state)를 가짐.
"""
import torch
import torch.nn as nn

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

"""
    - 모델을 sequence of layers으로 정의
    - nn.Sequential은 다른 Moudle들을 포함하는 Module
        - Module들을 순차적으로 적용하여 출력 생성
    - 각 Linear module은 선형 함수를 사용하여 입력으로부터 출력을 계산하고, 내부 Tensor에 가중치와 편향 저장
"""
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)

"""
    - nn 패키지는 손실 함수에 대한 정의도 포함됨.
    - 'sum' 은 출력 합산
    - 'mean'은 출력의 합이 출력의 요소 수로 나뉨.
"""
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # forward : 모델에 x 전달, 예상되는 y값 계산
    y_pred = model(x)
    
    # calc loss
    # predict y, label y를 갖는 Tensor들을 전달
    # 손실 값을 갖는 Tensor 반환
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # backward 전, 변화도 0으로
    model.zero_grad()
    
    """
    Backward
        - 아래 호출은 모델의 학습 가능한 모든 매개변수에 대한 손실 변화도 계산
        - 각 Module의 매개변수는 requires_grad=True 일 때 Tensor 내에 저장
    """
    loss.backward()
    
    # update weight using gradient descent
    # 각 매개변수는 Tensor이므로, 변화도에 접근 가능
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
# %%
"""
PyTorch: optim
    - 최적화 알고리즘에 대한 아이디어를 추상화하고 일반적으로 사용하는 최적화 알고리즘의 구현체 제공
"""

import torch
import torch.nn as nn

# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # forward
    y_pred = model(x)
    
    # calc loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    """
        - backward 전, optimizer 객체를 사용하여 갱신할 변수들의 변화도를 0으로 만듦.
        - backward()를 호출할 때마다 변화도가 누적되기 때문
    """
    optimizer.zero_grad()
    
    # backward: 매개변수에 대한 loss의 변화도 계산
    loss.backward()
    
    # optimizer의 step을 호출하면 매개변수가 갱신됨.
    optimizer.step()
# %%
"""
user defined nn.Module
    - 기존 모듈의 구성보다 복잡한 모듈을 구성할 때
    - nn.module의 서브클래스로 새 모듈 정의
    - 입력 Tensor를 받아 다른 모듈 또는 Tensor의 autograd 연산을 사용하여 출력 Tensor를 만드는 forward 정의
"""
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        2개의 nn.Linear 모듈 생성 후, 멤버 변수로 지정
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        """
        - 입력 데이터의 Tensor를 받고 출력 데이터의 Tensor를 반환
        - Tensor 상의 임의의 연산자뿐만 아니라 생성자에서 정의한 Module도 사용 가능
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)

# loss function
criterion = nn.MSELoss(reduction='sum')
# optimizer
# 생성자에 model.parameters()를 호출하면 모델의 멤버인 2개의 nn.Linear 모듈의 학습 가능한 매개변수들이 포함됨.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # forward
    y_pred = model(x)

    # calc loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도 0, backward 수행, 가중치 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# %%
"""
PyTorch: Control Flow + Weight Sharing
    - 일반적인 Python 제어 흐름을 사용하여 반복(loop)을 구현 가능
    - forward 정의 시, 단지 동일한 Module을 여러번 재사용함으로써 내부(innermost) 계층들 간의 가중치 공유 구현 가능
"""

import random
import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        생성자에서 forward에서 사용할 3개의 nn.Linear 인스턴스를 생성합니다.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = nn.Linear(D_in, H)
        self.middle_linear = nn.Linear(H, H)
        self.output_linear = nn.Linear(H, D_out)

    def forward(self, x):
        """
        모델의 forward에서, 무작위로 0, 1, 2 또는 3 중에 하나를 선택하고
        은닉층을 계산하기 위해 여러번 사용한 middle_linear Module을 재사용

        각 forward는 동적 연산 그래프를 구성하기 때문에, 모델의 forward를
        정의할 때 반복문이나 조건문과 같은 일반적인 Python 제어 흐름 연산자 사용 가능

        연산 그래프를 정의할 때 동일 Module을 여러번 재사용하는 것이 완벽히 안전하다는 것을 알 수 있음. 
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# 좌측부터, 배치 크기, 입력 차원, 은닉층 차원, 출력 차원
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

# 손실함수와 Optimizer를 만듭니다. 
# 이 모델을 순수한 확률적 경사 하강법 (stochastic gradient decent)으로 학습하는 것은 어려우므로, 모멘텀(momentum) 사용
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # forward
    y_pred = model(x)

    # calc loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도를 0으로 , backward 수행, 가중치 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()