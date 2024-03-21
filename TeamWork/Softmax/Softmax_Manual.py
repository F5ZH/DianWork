import torch

'''
引入IPython的display模块，这个模块可以在Jupyter上显示数据，例如图片。
'''
from IPython import display
from d2l import torch as d2l

batch_size = 256
'''
这是一个d2l库中已经编写完成的一个函数，包含了引入Fashion-MNIST数据集，
并且按照所给批量参数batch_size随机从数据库中调用一批数据作为训练集和测试集
其中还有一个可选参数resize，可以用来调整图片大小
'''
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
'''
torch.normal生成一个满足正态分布的随机数，其中包括三个必选参数，
分别是均值，标准差和输出张量的形状
可选参数requires_grad = True说明希望能对得到的数据进行梯度优化
'''
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
'''
生成一个十列一行的偏置，希望进行梯度优化
'''
b = torch.zeros(num_outputs, requires_grad=True)

'''
建立输出y与独热编码的联系，,使所有输出总和唯一，便于借用独热编码来找到最符合的项
采用按行求和，最后得到的张量每行的第一列为这行总和，每行对应一种输出y，
因此选用按行求和
另外，这个函数没有采取有效措施防止极大或者极小的数据在进行运算时发生数据溢出
需要注意。
'''


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


'''
建立了一个从特征张量到输出的映射，即包装了softmax处理的过程
'''


def net(X):
    '''
    这里使用reshape方法将x的形状(batch_size, 1, 28, 28)变成了一批线性张量(batch_size, 784)
    经过矩阵乘法之后得到了(batch_size, 10)
    然后根据广播机制将b于XW相加，得到一个(batch_size, 10)的张量
    '''
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


'''
定义交叉熵损失函数，这个函数多用于处理分类问题
当len函数作用于一个张量时，返回他的第一个维度的长度
在这里，用range(len(y_hat))生成的序列作行索引逐个检索每个样本，
用y中的一维张量找出真实对应的独热编码的数据，并取对取负，形成一维张量返回
'''


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


'''
测量分类精度，用于监测函数的准确性
'''


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        '''
        argmax函数的作用是返回一个数组中最大值的索引。
        在这里由于参数axis设置为1，即找出每行的最大值的序号
        '''
        y_hat = y_hat.argmax(axis=1)
        '''
        cmp返回一个布尔类型的一维张量，长度和y相同
        并且用type函数保证y与y_hat的数据类型保持相同，保证判断过程不会出错
        '''
    cmp = y_hat.type(y.dtype) == y
    '''
    由于sum中没有选择参数保证原张量的形状不变，
    最后得到的总和数据转化为float类型表示这组样本中测量结果正确的总数
    转化为float类型便于后续进行操作accuracy(y_hat, y) / len(y)得到测量精度
    '''
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    '''
    isinstance(net, torch.nn.Module)这行代码的作用是检查变量net是否是torch.nn.Module类的实例。
    一般情况下，我们更加建议这个神经网络是一个nn.Module的子类，因为这个时候他可以直接应用pytorch内部的train和eval方法
    这使得在函数运行时更加高效也更加便利，能让神经网络更加清楚知道目前的状态（训练或是评估）
    '''
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式 很重要！！！
        '''
        Accumulator(n)通常用于创建累加器对象以存储和更新累积值。
        其中参数n指明了要累计的值的数量，本例中是创建了两个累计值，分别为正确预测数、预测总数
        另外，在这里，这个类由书作者自己创建，但是实际可以从torch中直接使用
        '''
    metric = Accumulator(2)
    '''
    在测试精度时保持原有的参数不改变，因此不需要进行梯度计算
    同时也为程序节省了时间和空间
    '''
    with torch.no_grad():
        for X, y in data_iter:
            '''
            数据迭代器中包含了测试特征X与测试标签y（对应真实类别在这一独热编码中的序号）
            该循环每次截取一个批次对的张量（按照迭代器的定义），使用net神经网络得到预测值，
            通过accuracy函数来得到正确判断的个数，分别将这个数据与样本的总批次进行累计
            在返回函数时就得到了一个总体的精度
            '''
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


'''
创建了一个accumulator类，其中包含了四种方法
用于对累计值的计算
'''


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    '''
    如果是一个nn.Module类的神经网络，那么可以适用train方法
    '''
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            '''
            清空梯度，即调用updater.zero_grad()方法。这是为了确保在每次迭代时不会累积之前的梯度。
            '''
            updater.zero_grad()
            l.mean().backward()
            '''
            更新模型参数，即调用updater.step()方法。这将根据优化器的设置更新模型的权重和偏置。
            '''
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            '''
            使用自定义的优化器更新模型参数，即调用updater(X.shape[0])方法。
            我们期望这里自定义的优化器根据批量大小对参数更新。这里的X.shape[0]表示当前批次的样本数。
            '''
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失（所有样本的平均损失）和训练精度,返回的是一个元组
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  # @save
    """在动画中绘制数据"""

    '''
    这是类的构造函数，用于初始化对象的属性。
    参数包括x轴标签、y轴标签、图例、x轴范围、y轴范围、x轴刻度类型、y轴刻度类型、线条格式、子图行数、子图列数和图形大小等。
    '''

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        '''
        单次优化函数返回的是一个元组，返回了训练损失和训练精度
        '''
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        '''
        Python中，当只有一个元素的元组时，需要在元素后面加一个逗号来表示这是一个元组。
        因此，(test_acc,)表示一个只包含test_acc的元组。
        在这个代码中，train_metrics是一个元组，通过使用+操作符将两个元组连接起来，形成一个新的元组。
        所以，(test_acc,)这个逗号内没有东西接着是因为它本身就是一个元组，不需要再添加其他元素。
        '''
        animator.add(epoch + 1, train_metrics + (test_acc,))
        '''
        如上所说，train_metrics是一个含有两个元素的元组，在这里他的两个元素分别赋予train_loss和train_acc，
        对应训练损失和训练精度。
        '''
    train_loss, train_acc = train_metrics
    '''
    断言语句，用于检测训练函数是否符合标准
    如果不符合断言条件，则抛出一个异常并且输出当且的训练损失/训练精度/测试精度
    '''
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1

def updater(batch_size):
    '''
    使用随机梯度下降（SGD）算法来更新模型参数W和b。
    d2l.sgd是一个用于实现随机梯度下降的函数。
    它接受三个参数：需要更新的参数列表、学习率和批量大小。
    在这个例子中，参数列表为[W, b]，学习率为lr，批量大小为batch_size。
    '''
    return d2l.sgd([W, b], lr, batch_size)


'''
现在，我们训练模型10个迭代周期。 
请注意，迭代周期（）和学习率（）都是可调节的超参数。 通过更改它们的值，我们可以提高模型的分类精度。
'''
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    '''
    这段代码中的zip函数用于将两个列表（trues和preds）中的元素一一对应地组合成元组，
    然后通过列表推导式生成一个新的列表titles。
    具体来说，zip(trues, preds)会返回一个迭代器，其中每个元素都是一个包含两个元素的元组，
    第一个元素来自trues，第二个元素来自preds。
    通过列表推导式，我们可以将这些元组转换为字符串，并用换行符连接真实标签和预测标签。
    zip函数通常用于同时迭代两个或多个可迭代对象（如列表、元组等），当需要将来自不同可迭代对象的元素一一对应地组合时，zip函数非常有用。
    '''
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)