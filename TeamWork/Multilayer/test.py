import torch
from d2l import torch as d2l


'''
创建了一个张量（tensor）x，它包含了从-8.0到8.0（不包括8.0）的等间隔值，步长为0.1。
requires_grad=True表示需要计算x的梯度，这对于自动求导和反向传播非常重要。
'''
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
'''
应用了ReLU（Rectified Linear Unit）激活函数，将输入张量x中的负数置为0，保留正数不变。
ReLU函数常用于神经网络中，因为它可以引入非线性，并且计算效率高。
'''
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

'''
调用函数实例y的backward方法，进行梯度计算，并将其存入x.grad中
其中第二个参数值要求保留计算图，便于后续计算
'''
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 清除以前的梯度。在多任务学习或者模型复用时，清除梯度可以避免不同任务之间的梯度更新相互干扰。
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# 清除以前的梯度 在多任务学习或者模型复用时，清除梯度可以避免不同任务之间的梯度更新相互干扰。
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

