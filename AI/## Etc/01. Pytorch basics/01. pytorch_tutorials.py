#%%
from __future__ import print_function
import torch
#%%
x = torch.empty(5, 3)
print(x)
# %%
x = torch.rand(5, 3)
print(x)
# %%
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# %%
x = torch.tensor([5.5, 3])
print(x)
# %%
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
# %%
x = torch.randn_like(x, dtype=torch.float)
print(x)       
# %%
print(x.size())
# %%
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# %%
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# %%
y.add_(x)
print(y)
# %%
print(x[:, 1])
# %%
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
# %%
x = torch.randn(1)
print(x)
print(x.item())
# %%
a = torch.ones(5)
print(a)
# %%
b = a.numpy()
print(b)
# %%
a.add_(1)
print(a)
print(b)
# %%
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# %%
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA 장치 객체(device object)로
    y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
    x = x.to(device)                       # ``.to("cuda")`` 를 사용하면 됩니다.
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!
# %%
