import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np

batch = 10
high = 28
width = 28
epochs = 1


class NumberCnn(nn.Module):
    def __init__(self) -> None:
        """ 网络模型的定义 """
        super().__init__()
        # 输入为28*28=784维数据  降维到64维
        self.fc1 = nn.Linear(high * width, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # 最后输出10个特征的维度
        self.fc4 = nn.Linear(64, 10)

    def forward(self, z):
        """ 向前预测，通过网络的一系列操作 """
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.relu(self.fc2(z))
        z = nn.functional.relu(self.fc3(z))
        # Softmax对数函数最终通过四层输出10个类别的对数概率
        z = nn.functional.log_softmax(self.fc4(z), dim=1)
        return z


class Data_helper:
    @staticmethod
    def data_loader(is_train):
        # 组合多个转换操作，PIL图像numpyarray转Torch-Tensor, transforms.Normalize((0.5,), (0.5,))
        to_tensor = transforms.Compose([transforms.ToTensor()])
        # MNIST数据集实例，是否加载训练集，训练集加载时的转换操作，不存在数据集则下载
        data_set = MNIST("", is_train, transform=to_tensor, download=True)
        # DataLoader的数据加载器实例，训练或测试的数据加载，每个批次含样本数15，
        # 训练开始时打乱数据集顺序(随机化)，防止模型过拟合/提高泛化能力/确保各批次数据分布均匀
        load_data = DataLoader(data_set, batch_size=batch, shuffle=True)
        return load_data

    @staticmethod
    def evaluate(_data, net):
        """ 预测成功与样本总数量的比值 """
        n_correct, n_total = 0, 0
        # 只关心模型输出而不更新参数
        with torch.no_grad():
            # _in: 输入数据 _t: 标签
            for (_in, _t) in _data:
                # _in.view展平网络
                net_out = net.forward(_in.view(-1, high * width))
                for _n, _out in enumerate(net_out):
                    # 预测概率最大的类别所对应的标签
                    if torch.argmax(_out) == _t[_n]:
                        n_correct += 1
                    n_total += 1
        return n_correct / n_total


class Img_helper:
    @staticmethod
    def show_predict_view(inputs, lab, outputs):
        """ 训练集查看 """
        value, predicted = torch.max(outputs, 1)
        for index, sample in enumerate(lab):
            sample_input = inputs[index]
            # 将输入Tensor转换为NumPy数组，并显示图像
            plt.figure(index, figsize=(5, 5))
            plt.imshow(sample_input[0].numpy())
            plt.title(f"True label: {sample}\nPredicted label: {predicted}\n({value})")
        plt.show()


if __name__ == "__main__":
    ih = Img_helper()
    dh = Data_helper()
    cnn_model = NumberCnn()
    train_data = dh.data_loader(is_train=True)
    test_data = dh.data_loader(is_train=False)
    # 损失函数
    criterion = nn.NLLLoss()
    # 优化器，随机梯度下降（SGD）算法
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.003)

    '''核心的卷积神经网络 epoch 反复训练的轮次，为了提高数据集的利用率 '''
    for epoch in range(epochs):
        print(f"—— 第{epoch}次训练 ——")
        for n, (_v, _t) in enumerate(train_data):
            print(f"数据标签集{_t}")
            # 图像展平
            img = _v.view(batch, -1)
            # 清零所有模型的参数梯度，否则之前的backward会累积而不是替换
            optimizer.zero_grad()
            # 正向传播
            output = cnn_model(img)
            # 计算差值 nll_loss为对数损失函数(为了匹配log_softmax的对数运算)
            loss = nn.functional.nll_loss(output, _t)
            # 反向误差传播
            loss.backward()
            # 利用之前的梯度更新模型参数
            # 目前为随机梯度下降（SGD）算法，会用每个参数的梯度乘以学习率，然后从该参数的当前值中减去这个结果
            optimizer.step()
            # # 看看need效果
            # ih.show_predict_view(_v, _t, output)

        print(f"epoch {epoch} accuracy {dh.evaluate(test_data, cnn_model)}")

    ''' 以上训练结束，运行测试集 '''

    for n, (_v, _t) in enumerate(test_data):
        if n > 3:
            break
        img = _v[0].view(-1, high * width)
        output = cnn_model(img)

        ih.show_predict_view(_v, _t, output)

        """ 绘图 """
    #     data = _v[0].view(high, width)
    #     plt.figure(n, figsize=(10, 10))
    #     plt.imshow(data)
    #     # 在每个像素点上添加注释
    #     arr = data.numpy()
    #     for i in range(arr.shape[0]):
    #         for j in range(arr.shape[1]):
    #             text = plt.text(j, i, np.around(arr[i, j], decimals=2), ha="center", va="center",
    #                             color="white", fontsize=7, fontweight='bold')
    # plt.show()
