"""
- 이미지는 Pillow나 OpenCV 같은 패키지
- 오디오를 처리할 때는 SciPy와 LibROSA
- 텍스트의 경우에는 그냥 Python이나 Cython 사용, NLTK나 SpaCy도 유용
"""
#%%
import torch
from torch.utils.data.dataset import BufferedShuffleDataset
import torchvision
import torchvision.transforms as transforms
#%%
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# %%
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# %%
""" CNN define """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# %%
""" loss function, Optimizer """
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# %%
""" Nerual Network Learning """
for epoch in range(2) : # iterate dataset
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data   # data = [inputs, labels]
        
        optimizer.zero_grad()   # grad = 0
        
        # forward + backward + optimization
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# %%
""" save the model """
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
# %%
dataiter = iter(testloader)
images, labels = dataiter.next()

""" GroundTruth : 진짜 정답과 예측을 비교하는 방식 """
# 이미지 출력
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join
        ('%5s' % classes[labels[j]] for j in range(4)))
# %%
# model load
net = Net()
net.load_state_dict(torch.load(PATH))
# %%
outputs = net(images)
# %%
"""
    - 어떤 분류에 대해 더 높은 값이 나타난다는 것은 그 이미지가
    - 해당 분류에 더 가깝다고 생각하는 것임.
"""
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
# %%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
# %%
"""
0. for i in range(10)
- 0.0 10번
"""
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():   # 내부 텐서의 requires_grad = False로 설정
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()     # squeeze는 1차원인 축 제거
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치 출력
print(device)
# %%
# 모든 모듈의 매개변수와 버퍼를 CUDA tensor로 변경
net.to(device)
# 각 단계에서 입력과, 정답도 GPU로 보내야 함.
inputs, labels = data[0].to(device), data[1].to(device)