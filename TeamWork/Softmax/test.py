import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


'''  使图形以SVG的格式显示   '''
d2l.use_svg_display()


# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
# 并会将图像的维度从(H, W, C)转换为(C, H, W)，其中H是高度，W是宽度，C是通道数。
trans = transforms.ToTensor()
'''
这行代码加载了Fashion-MNIST数据集的训练集。
torchvision.datasets.FashionMNIST 是torchvision提供的用于加载Fashion-MNIST数据集的类。
root="../data" 参数指定了数据集的下载位置，train=True 参数告诉torchvision加载的是训练集。
transform=trans 参数指定了对加载的图像进行哪种转换操作，这里就是之前创建的ToTensor转换。
download=True 参数表示如果数据集不在指定的root目录中，则从互联网上下载数据集。
'''
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
'''
train=False表示加载的为测试集，不用于对模型的训练
'''
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)


'''
获取训练集和测试集的样本数量（这里仅供展示）
'''
print(len(mnist_train), len(mnist_test))


'''
同样为展示，在PyTorch中，torchvision.datasets.FashionMNIST数据集的每个元素是一个元组，
其中第一个元素是图像，第二个元素是相应的标签y。
图像本身是一个PyTorch张量（torch.Tensor）。
mnist_train[0] 获取训练集 mnist_train 中的第一个样本（图像和标签的元组）。
mnist_train[0][0] 从这个元组中获取第一个元素，即图像。
.shape 是张量（Tensor）的一个属性，返回张量的维度大小。
在Fashion-MNIST数据集中，每个图像是一个28x28像素的单通道（黑白）图像。
因此，mnist_train[0][0].shape 的输出应该是 (1, 28, 28)，表示这是一个单通道图像，高度为28像素，宽度也为28像素。
注意，第一个维度的大小为1，表示单通道。
'''
print(mnist_train[0][0].shape)


'''
用于实现序号与对应实际标签的转化
'''
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    '''
    通过scale参数按比例调整图像大小
    '''
    figsize = (num_cols * scale, num_rows * scale)
    '''
    这行代码使用 d2l.plt.subplots 函数创建了一个图形和一组轴（axes）。
    num_rows 和 num_cols 指定了轴的布局，figsize 指定了图形的大小。
    _ 表示图形对象的占位符，axes 是一个包含了所有轴的数组。
    '''
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    '''
    将 axes 数组展平成一个一维数组，这样就可以轻松地遍历每个轴。
    '''
    axes = axes.flatten()
    '''
    zip函数将轴与图像配对
    '''
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        '''
        将图像数据（张量或者PIL文件）处理成同一类型后进行显示
        '''
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
            '''
            隐藏坐标轴
            '''
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        '''
        如果提供了标题列表，将为每个轴设置标题
        '''
        if titles:
            ax.set_title(titles[i])
            '''
            返回轴的数组，便于后续操作
            '''
    return axes


'''
data.DataLoader：这是PyTorch中的一个类，用于创建一个数据加载器，它可以从数据集中提供批量数据。
数据加载器非常有用，因为它们可以在训练循环中高效地加载数据，并且可以自动处理数据并行和混洗（shuffle）。
mnist_train：这是之前加载的Fashion-MNIST训练集。
batch_size=18：这指定了每个批次中包含的样本数量。
iter(data.DataLoader(...))：这里创建了一个数据加载器的迭代器。
next(...)：这获取了迭代器的下一个元素，即一个批次的数据。这个批次包含18个样本，每个样本包括一个图像（X）和一个标签（y）。
'''
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
'''
显示图片（需要pyplot库）
'''
plt.show()


'''
读取小批量数据（更高效的方法）
'''
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())


'''
测试读取训练数据运行时间
'''
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'


'''
定义load_data_fashion_mnist函数，用于获取和读取Fashion-MNIST数据集。 
这个函数返回训练集和验证集的数据迭代器。 此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。
'''
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    '''
    trans = [transforms.ToTensor()] 这行代码初始化了一个转换列表 trans，并添加了一个转换操作 transforms.ToTensor()。
    这个转换操作会将PIL图像或NumPy数组转换为PyTorch张量。
    resize 参数是一个整数，表示新的图像尺寸，例如 resize=256 将会将图像尺寸调整为256x256像素。
    如果设置了 resize 参数，这行代码会将一个新的转换操作 transforms.Resize(resize) 插入到转换列表 trans 的开头。
    insert(0, ...) 方法将元素插入到列表的第一个位置。transforms.Resize(resize) 是一个将图像尺寸调整为指定大小的转换操作。
    trans = transforms.Compose(trans) 这行代码使用 transforms.Compose 函数将转换列表 trans 组合成一个单一的转换操作。
    transforms.Compose 接受一个转换操作的列表，并按照列表的顺序应用这些转换操作。
    '''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


'''
x， y分别读取训练集中的特征与标签
'''
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

    