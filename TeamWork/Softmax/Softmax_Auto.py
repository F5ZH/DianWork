import torch
from torch import nn
from d2l import torch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
'''
n.Sequential 在 PyTorch 中是一个容器类，它按照层添加的顺序依次执行每个层的操作。
nn.Flatten 是一个简单层，没有权重和偏置。它的作用是将输入数据展平为一维数组。
这个操作不涉及任何计算，只是改变了数据的形状。
nn.Linear 会在内部产生权重 W 和偏置 b，并允许它们进行梯度优化。
它接受两个参数：输入特征的数量和输出特征的数量。这个层的权重矩阵 W 的形状是 (输出特征数， 输入特征数)，偏置向量 b 的形状是 (输出特征数，)。
当数据通过 nn.Linear 层时，会发生Y = XW + b
无论是 nn.Linear 还是 nn.Flatten，它们都设计为处理批量数据，因此在处理输入张量时会保持行数不变，即保持样本数量不变。
'''
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        '''
        nn.init.normal_()函数的语法如下：torch.nn.init.normal_(tensor, mean=0, std=1)
        tensor是要初始化的张量，mean表示正态分布的均值（默认为0），std表示正态分布的标准差（默认为1）。
        这里将m层的权重值初始化为均值为0、标准差为0.01的正态分布随机值。
        '''
        nn.init.normal_(m.weight, std=0.01)

'''
对net的所有层应用这个函数
'''
net.apply(init_weights)

'''
赋予一个交叉熵损失函数的实例
'''
loss = nn.CrossEntropyLoss(reduction='none')


'''
这行代码创建了一个随机梯度下降（SGD）优化器对象，并将其赋值给变量trainer。
torch.optim.SGD 类初始化时需要的参数主要包括：
params：这是一个必须提供的参数，它是一个包含需要优化的参数（张量）的迭代器
在这里net的参数为权重值和偏置两种张量
lr：学习率（learning rate），这也是一个必须提供的参数。它是一个正数，控制每次参数更新的步长。学习率的大小会影响训练的速度和效果，
'''
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)