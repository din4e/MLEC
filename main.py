import torch
import copy
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
LR = 0.01  # 学习率
N = 16  # 节点数目
zero = torch.zeros(BATCH_SIZE, N)
one = torch.ones(BATCH_SIZE, N)
train_data_path = r'dataset/traindata500.txt'
test_data_path = r'dataset/testdata500.txt'


def getLambda(a = [[]], x = []) -> float:
    _a = copy.deepcopy(a)
    n = len(a)
    if n != 0:
        if len(x)!=0:
            for i in range(n):
                if x[i] == 1:
                    for j in range(n):
                        _a[i][j] = _a[j][i] = 0
        b, _ = np.linalg.eig(_a)
        Lambda = max(b)
    else:
        return 0.0
    return Lambda

class Graph:
    def __init__(self, n, a=[[]]):
        self.N = n
        self.a = copy.deepcopy(a)
        self.vertices = []
        for i in range(n):
            self.vertices.append([]);
        self.edges = []
        self.getVerticeAndEdge(a)
        self.Lambda = getLambda(a)

    def size(self) -> int:
        return self.N

    def getVerticeAndEdge(self, a):
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                if a[i][j] == 1:
                    self.vertices[i].append(j)
                    self.vertices[j].append(i)
                    self.edges.append([i,j])

    def printV(self):
        print("Vertices:")
        for i in range(self.N):
            print("Node", i, end='')
            for v in self.vertices[i]:
                print(v, end=' ')
            print()

    def printE(self):
        print("Edges:")
        for e in self.edges:
            print(e[0], e[1])

    def print(self):
        self.printV()
        self.printE()


class Strategy:
    def __init__(self, index = 0,x = [], Lambda = 0.0):
        self.NEtype = ['NotNE', 'NE', 'MinNE', 'MaxNE']
        self.NEindex = index
        x = np.array(x)
        self.N = x.shape[0]
        self.X = []  # np.empty(shape=[0, self.N])
        self.X.append(x)  # 枚举算法可能有多个策略
        self.Cost = 0
        for i in x:
            if i == 1:
                self.Cost = self.Cost + 1
        self.Lambda = []
        self.Lambda.append(Lambda)  # 枚举算法可能有多个策略 对应多个lambda值

    def print(self):
        print("Type:%s Cost:%d" % (self.NEtype[self.NEindex], self.Cost))
        for i, x in enumerate(self.X):
            print("Strategy %d (Lambda %f): Secured " % (i+1, self.Lambda[i]), end='')
            for j, _x in enumerate(x):
                if _x == 1:
                    print(j, end=' ')
                else:
                    _ # print(j, end=' ')
            print()

class EC:
    # a:numpy
    def __init__(self, a, T:float):
        self.N = a.shape[0]  # tensor -> list
        self.G = copy.deepcopy(Graph(self.N, a))
        self.a = copy.deepcopy(self.G.a)
        self.Lambda = 0.0
        self.T = T

    def getLambda(self, a = [], x = []) -> float:
        # print(self.G.a)
        if len(a) == 0:
            self.a = copy.deepcopy(self.G.a)
        if len(x) != 0:
            for i in range(len(x)):
                if x[i] == 1:
                    # print(i)
                    for j in range(self.N):
                        self.a[i][j] = self.a[j][i] = 0
        b, _ = np.linalg.eig(self.a)
        self.Lambda = max(b)
        return self.Lambda

    def isNE(self, x) -> bool:
        return

    def iterativesecure(self, pi, rho=[]):
        x = [0 for _ in range(self.N)]
        # Lambda = self.getLambda([],x)
        # print(Lambda)
        if len(pi) != self.N:
            return [], -1.0
        if len(rho) == 0:
            rho = list(reversed(pi))
        for p in pi:
            x[p] = 1  # noerror print(p,x)
            if self.getLambda([], x) < self.T:
                break  # print(x, self.getLambda([], x),self.T,'/')
        for r in rho:
            if x[r] == 1:
                x[r] = 0
                x[r] = 1 if self.getLambda([], x) >= self.T else 0 # print(x, self.getLambda([], x), self.T)
        # print(x)
        return x, self.getLambda([], x)

    def HDG(self):
        l = []
        for i,v in enumerate(self.G.vertices):
            l.append([i,len(v)])
        l.sort(key = lambda x:x[1], reverse = True )
        pi = [x[0] for x in l]
        rho = list(reversed(pi))
        x, Lambda = self.iterativesecure(pi, rho)
        return Strategy(2, x, Lambda)

    def LDG(self):
        return


class ECDataset(Dataset):
    data: Tensor
    label: Tensor

    def __init__(self, txt_path, is_traindata=True, transform=None, target_transform=None):
        self.is_traindata = is_traindata
        self.transform = transform
        self.target_transform = target_transform
        # _data = np.loadtxt(txt_path, dtype=float, delimiter=',')
        _data = np.genfromtxt(txt_path, delimiter=',', dtype=float)
        # _data = np.delete(_data, -1, axis=1)
        _data, _label = np.split(_data, [N * N], 1)

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
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)  # 10*10->9*9
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)  # 9*9 -> 8*8
        self.line1 = nn.Linear((N - 2) * (N - 2), N)

    def forward(self, xb):
        xb = xb.view(-1, 1, N, N)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = self.line1(xb.view(xb.size(0), -1))
        return xb


if __name__ == '__main__':
    # a = np.array([[1, 2, 3], [2, 3, 4], [2, 1, 3]])
    # b, _ = np.linalg.eig(a)
    # b = max(b)
    # print(b)

    train_data = ECDataset(train_data_path, True)

    for i in range(len(train_data)):
        a, y = train_data[i]
        # a, _ =   # 根据traindata/testdata获取邻接矩阵
        G = Graph(N, a)  # 根据邻接矩阵a 获得图的点边信息
        ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
        s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
        # y=int(y)
        y = y.int().numpy()
        cnt_y = 0
        for i in y:
            if i == 1:
                cnt_y = cnt_y + 1
        cnt_hdg = 0
        for i in s.X[0]:
            if i == 1:
                cnt_hdg = cnt_hdg + 1
        # print("1: ", y, cnt_y)
        # print("2: ", s.X[0], cnt_hdg) # , end=' ')
        print(cnt_y-cnt_hdg)
        # s.print()


    # a = a.numpy()
    # b = copy.deepcopy(a)
    # print(a is b)
    # x = np.array([[], [], [11]])
    # b = [ [1,1], [1,1]]
    # x = [1,1,1,1, 1,1,1,1, 0,0,0,0, 1,1,1,1]
    # print(getLambda(a, x)) # print(any(x))

    # print(x.shape[0])
    # print("数组的维度数目", a.shape[0])
    # print(a)
    a, _ = train_data[0]
    # G = Graph(N, a)  #  根据邻接矩阵a 获得图的点边信息
    # # print(G.Lambda)
    # ec = EC(a, G.Lambda*0.5)  #  根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
    # s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
    # s.print()
    exit()

    test_data = ECDataset(test_data_path, False)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    loss_count = []
    acc_count = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    model = CNN().double()  # 把数据都转成double 否则跑不起来
    loss_func = nn.MSELoss()  # 损失函数用的 平方差公式
    opt = optim.Adam(model.parameters(), lr=LR)

    # opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = loss_func(out, y)  # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上

            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())  # torch.save(model, r'')
            acc = []
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
