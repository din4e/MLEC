import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

EPOCH = 50
BATCH_SIZE = 100
LR = 0.01
N = 16
zero = torch.zeros(BATCH_SIZE, N)
one = torch.ones(BATCH_SIZE, N)
train_data_path = r'dataset/traindata500.txt'
test_data_path = r'dataset/testdata500.txt'

class ECDataset(Dataset):
    data: Tensor
    label: Tensor

    def __init__(self, txt_path, is_traindata = True, transform = None, target_transform = None):
        self.is_traindata = is_traindata
        self.transform = transform
        self.target_transform = target_transform
        # _data = np.loadtxt(txt_path, dtype=float, delimiter=',')
        _data = np.genfromtxt(txt_path, delimiter=',', dtype=float)
        # _data = np.delete(_data, -1, axis=1)
        _data, _label = np.split(_data, [N*N], 1)

        self.data = torch.from_numpy(_data).double()
        self.data = self.data.view(-1, N, N)
        self.label = torch.from_numpy(_label).double()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.size(0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0) # 10*10->9*9
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0) # 9*9 -> 8*8
        self.line1 = nn.Linear((N-2)*(N-2), N)
    def forward(self, xb):
        xb = xb.view(-1, 1, N, N)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = self.line1(xb.view(xb.size(0), -1))
        return xb

class EC:
    # a:numpy
    def getLambda(self):
        b, _ = np.linalg.eig(self.a)
        return max(b)
    def __init__(self,a):
        self.N = a.shape
        self.a = a
        self.Lambda = 0
    def HDG(self):
        return
    def isSecure(self):
        return

if __name__ == '__main__':
    # a = np.array([[1, 2, 3], [2, 3, 4], [2, 1, 3]])
    # b, _ = np.linalg.eig(a)
    # b = max(b)
    # print(b)

    # print("数组的维度数目", a.shape)
    # exit()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used
    model = CNN().double()     # !!!
    loss_func = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    # opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
    train_data = ECDataset(train_data_path,True)
    test_data = ECDataset(test_data_path,False)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=True)
    loss_count = []
    acc_count = []

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = loss_func(out, y)  # 使用优化器优化损失
            opt.zero_grad()           # 清空上一步残余更新参数值
            loss.backward()           # 误差反向传播，计算参数更新值
            opt.step()                # 将参数更新值施加到net的parmeters上

            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item()) # torch.save(model, r'')
            acc=[]
            for test_x, test_y in test_loader:
                # test_x = Variable(a)
                # test_y = Variable(b)
                _y = model(test_x)
                _y = torch.where(_y > 0.5, one, zero)
                accuracy = _y.numpy() == test_y.numpy()
                acc.append(accuracy.mean())
            # print(acc)
            acc = np.array(acc)
            acc_count.append(acc.mean())
            print('accuracy:\t', acc.mean())

    plt.figure('Result')
    plt.plot(loss_count, label='Loss')
    plt.plot(acc_count, label='Acc')
    plt.legend()
    plt.savefig("loss500.png")
    plt.show()