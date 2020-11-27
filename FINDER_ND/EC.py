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
import time
import math
from getL import getL

EPOCH = 5           
BATCH_SIZE = 100
LR = 0.01  # 学习率
N = 20  # 节点数目
zero = torch.zeros(BATCH_SIZE, N)
one = torch.ones(BATCH_SIZE, N)
train_data_path = r'dataset/traindata10000_20hdg.txt'
test_data_path  = r'dataset/testdata10000_20hdg.txt'
figure_name     = r'10000_20hdg.png'

# def getL(a)->float:
#     N = len(a)
#     k = 0
#     X = [0.0 for _ in range(N)]
#     Y = [1.0 for _ in range(N)]
#     max1 = 0.0
#     max2 = float(N)
#     m1 = 0.0 
#     m2 = 0.0
#     tmpfloat = 0.0
#     while ((math.fabs(max1 - max2) > 0.001 ) or ( k < 3)):
#         max2 = max1
#         m1 = m2
#         for i in range(N):
#             X[i] = 0.0
#             for j in range(N): 
#                 X[i] += a[i][j] * Y[j];
#         tmpfloat = 0.0
#         for x in X:
#             if math.fabs(x) > tmpfloat: 
#                 tmpfloat = x 
#         if tmpfloat == 0.0:
#             return  0.0
#         else:
#             for i in range(N):
#                 Y[i] = X[i] / tmpfloat;
#             m2 = tmpfloat
#             max1 = math.sqrt(m1*m2)
#             if k > 3000:
#                 break
#             else: 
#                 k = k + 1
#     return max1

def getLambda(a = [[]], x = []) -> float:
    _a = copy.deepcopy(a)
    n = len(a)
    # t1 = time.time()
    if n != 0:
        if len(x)!=0:
            for i in range(n):
                if x[i] == 1 or x[i] == 1.0:
                    for j in range(n):
                        _a[i][j] = _a[j][i] = 0
        # b, _ = np.linalg.eig(_a)
        # Lambda = max(b)
        Lambda = getL(_a)
    else:
        return 0.0
    # t2 = time.time()
    # print(t2-t1)
    return Lambda

class ECGraph:
    def __init__(self, n, a=[[]]):
        self.N = n
        self.a = copy.deepcopy(a)
        self.vertices = []
        for _ in range(n):
            self.vertices.append([]);
        self.edges = []
        self.getVerticeAndEdge(a)
        self.Lambda = getLambda(a)

    def size(self) -> int:
        return self.N

    def getVerticeAndEdge(self, a):
        # print(self.N)
        for i in range(self.N-2):
            for j in range(i+1, self.N-1):
                if a[i][j] == 1 or a[i][j]==1.0:
                    self.vertices[i].append(j)
                    self.vertices[j].append(i)
                    self.edges.append([i,j])

    def printV(self):
        print("Vertices:")
        for i in range(self.N):
            print("Node", i, end=':')
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
        self.X = []       # np.empty(shape=[0, self.N])
        self.X.append(x)  # 枚举算法可能有多个策略
        self.Cost = 0
        for i in x:
            if i == 1 or i == 1.0:
                self.Cost = self.Cost + 1
        self.Lambda = []
        self.Lambda.append(Lambda)  # 枚举算法可能有多个策略 对应多个lambda值

    def print(self):
        print("Type:%s Cost:%d" % (self.NEtype[self.NEindex], self.Cost))
        for i, x in enumerate(self.X):
            print("Strategy %d (Lambda %f): Secured " % (i+1, self.Lambda[i]), end='')
            for j, _x in enumerate(x):
                if _x == 1 or _x == 1.0:
                    print(j, end=' ')
                else:
                    _ # print(j, end=' ')
            print()

class EC:
    def __init__(self, a, T:float):
        self.N = a.shape[0]  # tensor -> list
        self.G = copy.deepcopy(ECGraph(self.N, a))
        self.a = copy.deepcopy(self.G.a)
        self.Lambda = 0.0
        self.T = T

    def getLambda(self, a = [], x = []) -> float:
        # print(self.G.a)
        # t1 = time.time()
        if len(a) == 0:
            self.a = copy.deepcopy(self.G.a)
        # t2 = time.time()
        if len(x) != 0:
            for i in range(len(x)):
                if x[i] == 1 or x[i] == 1.0:
                    for j in range(self.N):
                        self.a[i][j] = self.a[j][i] = 0
        # t3 = time.time()
        b, _ = np.linalg.eig(self.a)
        # t4 = time.time()
        self.Lambda = max(b)

        # self.Lambda = getL(self.a)
        # print(self.Lambda,max(b))
        # t5 = time.time()
        # print(t4-t1,t5-t4)
        return self.Lambda

    def isNE(self, X) -> bool:
        if self.getLambda([],X)>=self.T:
            print("Total ERROR")
            return False
        for i,x in enumerate(X):
            if x == 1 or x == 1.0:
                X[i] = 0
                if self.getLambda([],X) < self.T:
                    print("enum ERROR")
                    return False
                X[i] = 1
        return True

    def iterativesecure(self, pi, rho=[]):
        x = [0 for _ in range(self.N)]
        #if len(pi) != self.N: # for FINDER is
        #    return [], -1.0
        # if len(rho) == 0:
        #     rho = list(reversed(pi))
        # print(pi)
        rho = []
        for p in pi:
            x[p] = 1
            rho.append(p)
            if self.getLambda([], x) < self.T:
                # print(x, self.getLambda([], x),self.T,'/')
                break  
        rho = list(reversed(rho))
        for r in rho:
            x[r] = 0
            # x[r] = 1 if self.getLambda([], x) >= self.T else 0 # print(x, self.getLambda([], x), self.T)
            if self.getLambda([], x) >= self.T:
                x[r] = 1
        return x, self.getLambda([], x)

    def FINDER(self,sol):
        x, Lambda = self.iterativesecure(sol)
        if not self.isNE(x):
            print("FINDER ERROR")
        return Strategy(2, x, Lambda)
    
    def FINDER_MAX(self,sol):
        sol = list(reversed(sol))
        x, Lambda = self.iterativesecure(sol)
        if not self.isNE(x):
            print("FINDER_MAX ERROR")
        return Strategy(3, x, Lambda)

    def HDG(self):
        l = []
        for i, v in enumerate(self.G.vertices):
            l.append([i, len(v)])
        l.sort(key = lambda x : x[1], reverse = True )
        pi = [x[0] for x in l]
        x, Lambda = self.iterativesecure(pi)
        if not self.isNE(x):
            print("HDG ERROR")
        return Strategy(2, x, Lambda)

    def LDG(self):
        l = []
        for i, v in enumerate(self.G.vertices):
            l.append([i, len(v)])
        l.sort(key = lambda x : x[1], reverse = False)
        pi = [x[0] for x in l]
        x, Lambda = self.iterativesecure(pi)
        if not self.isNE(x):
            print("LDG ERROR")
        return Strategy(3, x, Lambda)


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
        _data, _label = np.split(_data, [N * N], 1)

        self.data = torch.from_numpy(_data).double()
        self.data = self.data.view(-1, N, N)
        self.label = torch.from_numpy(_label).double()

        # self.hdg = []
        # for data in self.data:
        #     a = data.numpy()
        #     G = ECGraph(N, a)
        #     ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
        #     s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
        #     self.hdg.append(s.X)


    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.size(0)

def getHDG(x):
    hdglist = []
    for data in x:
        a = data.numpy()
        G = ECGraph(N, a)
        ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
        s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
        hdglist.append(s.X)
    return np.array(hdglist).reshape(BATCH_SIZE, N)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)  # N*N->(N-1)*(N-1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)  # (N-1)*(N-1)->(N-2)*(N-2)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)
        self.line1 = nn.Linear((N - 2) * (N - 2), N)

        # self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)  # N*N->(N-2)*(N-2)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)  # (N-2)*(N-2) -> (N-4)*(N-4)
        # self.line1 = nn.Linear((N - 4) * (N - 4), N)

        # self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)  # N*N->(N-4)*(N-4)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)  # (N-4)*(N-4)-> (N-8)*(N-8)
        # self.line1 = nn.Linear((N - 8) * (N - 8), N)

    def forward(self, xb):
        # x = copy.deepcopy(xb)
        xb = xb.view(-1, 1, N, N)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = torch.tanh(self.line1(xb.view(xb.size(0), -1)))
        # xb = self.line1(xb.view(xb.size(0), -1))
        # print(xb)
        return xb
        # xb_is = []
        # for j, a in enumerate(x):
        #     l = []
        #     for k, v in enumerate(xb[j]):
        #         l.append([k, v])
        #     l.sort(key=lambda v: v[1], reverse=False)
        #     pi = [v[0] for v in l]
        #     rho = list(reversed(pi))
        #     G = ECGraph(N, a.numpy())  # 根据邻接矩阵a 获得图的点边信息
        #     ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
        #     xx, Lambda = ec.iterativesecure(pi, rho)  # Strategy([1, 0, 0, 1, 1])
        #     if not ec.isNE(xx):
        #         print("ERROR")
        #     xb_is.append(xx)
        # # print(xb_is[0], type(xb_is))
        # # print(xb[0], type(xb))
        # xb_is = torch.Tensor(xb_is)
        # xb_is = Variable(xb_is.double(), requires_grad=True)
        # return xb_is


if __name__ == '__main__':
    # getLambda
    # a = np.array([[1, 2, 3], [2, 3, 4], [2, 1, 3]])
    # b, _ = np.linalg.eig(a)
    # b = max(b)
    # print(b)

    train_data = ECDataset(train_data_path, True)

    # for i in range(len(train_data)):
    #     a, y = train_data[i]
    #     # a, _ =   # 根据traindata/testdata获取邻接矩阵
    #     G = ECGraph(N, a)  # 根据邻接矩阵a 获得图的点边信息
    #     ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
    #     s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
    #     # y=int(y)
    #     y = y.int().numpy()
    #     cnt_y = 0
    #     for j in y:
    #         if j == 1:
    #             cnt_y = cnt_y + 1
    #     cnt_hdg = 0
    #     for j in s.X[0]:
    #         if j == 1:
    #             cnt_hdg = cnt_hdg + 1
    #     # print("1: ", y, cnt_y)
    #     # print("2: ", s.X[0], cnt_hdg) # , end=' ')
    #     print(i)
    #     if cnt_hdg-cnt_y>=2:
    #         print(y,s.X[0])
    #     # s.print()

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
    # a, _ = train_data[0]
    # G = ECGraph(N, a)  #  根据邻接矩阵a 获得图的点边信息
    # # print(G.Lambda)
    # ec = EC(a, G.Lambda*0.5)  #  根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
    # s = ec.HDG()  # Strategy([1, 0, 0, 1, 1])
    # s.print()
    # exit()

    test_data = ECDataset(test_data_path, False)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    loss_count = []
    acc_count = []
    gap_count = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    model = CNN().double()  # 把数据都转成double 否则跑不起来
    loss_func = nn.MSELoss()  # 损失函数用的 平方差公式
    opt = optim.Adam(model.parameters(), lr=LR)
    # opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

    for epoch in range(EPOCH):
        print(" Epoch ", epoch)
        for i, (x, y) in enumerate(train_loader):
            # print(type(x), type(y))
            # print(y, getHDG(x))
            out = model(x)
            # hdglist = getHDG(x) # print(hdglist)
            # out_np = out.detach().numpy()
            # out_is=[]
            # # print(type(out))
            # for j, a in enumerate(x):
            #     l = []
            #     for k, v in enumerate(out_np[j]):
            #         l.append([k, v])
            #     # print(l)
            #     l.sort(key=lambda v: v[1], reverse=True)
            #     pi = [v[0] for v in l]
            #     rho = list(reversed(pi))
            #     G = ECGraph(N, a)  # 根据邻接矩阵a 获得图的点边信息
            #     ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
            #     xx, Lambda = ec.iterativesecure(pi, rho)  # Strategy([1, 0, 0, 1, 1])
            #     out_is.append(xx)
            # out_is = torch.from_numpy(np.array(out_is).reshape(BATCH_SIZE,N)).double

            loss = loss_func(out, y)  # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上

            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())  # torch.save(model, r'')

            gap = 0.0 # acc = []
            for test_x, test_y in test_loader:
                _y = model(test_x) # test_x = Variable(a) # test_y = Variable(b)
                _y_is = []
                for j, a in enumerate(test_x):
                    l = []
                    for k, v in enumerate(_y[j]):
                        l.append([k, v])
                    l.sort(key=lambda v: v[1], reverse=True)
                    pi = [v[0] for v in l]
                    rho = list(reversed(pi))
                    G = ECGraph(N, a.numpy())  # 根据邻接矩阵a 获得图的点边信息
                    ec = EC(a, G.Lambda * 0.5)  # 根据邻接矩阵a 获得图的点边信息 以及ECGame相关的参数
                    xx, Lambda = ec.iterativesecure(pi, rho)  # Strategy([1, 0, 0, 1, 1])
                    _y_is.append(xx)
                _y_is = torch.Tensor(_y_is)
                _y_is = Variable(_y_is.double(), requires_grad=True)
                # hdglist = getHDG(test_x)
                hdglist = test_y
                for j in range(len(_y)):
                    cnt_y = 0
                    cnt_hdg = 0
                    for k in _y_is[j]: # for k in _y_is[j]:
                        if k == 1 or k == 1.0:
                            cnt_y = cnt_y + 1
                    for k in hdglist[j]:
                        if k == 1 or k == 1.0:
                            cnt_hdg = cnt_hdg + 1
                    # print(cnt_y, cnt_hdg)
                    gap = gap+(cnt_y-cnt_hdg)
                    # _y = torch.where(_y > 0.5, one, zero)
                    # accuracy = _y.numpy() == test_y.numpy()
                    # acc.append(accuracy.mean())
                # print(gap)
            print("Gap", gap/len(test_data))
            gap_count.append(gap/len(test_data))

            # print(acc)
            # acc = np.array(acc)
            # acc_count.append(acc.mean())
            # print('accuracy:\t', acc.mean())
            # print(i," Gap ", gap)

    plt.figure('Result')
    plt.plot(loss_count, label='Loss')
    plt.plot(gap_count, label='Gap')
    plt.legend()
    plt.savefig(figure_name)
    plt.show()