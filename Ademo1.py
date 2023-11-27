# import torch
#
# # 输入张量x
# x = torch.randn(100, 1)
# # 输出张量y
# y = 3 * x + 2
#
# # 定义模型
# model = torch.nn.Linear(1, 1)
#
# # 使用均方误差作为损失函数，使用随机梯度下降作为优化器：
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # 训练模型
# for t in range(1000):
#     # 前向传播
#     y_pred = model(x)
#
#     # 计算损失
#     loss = loss_fn(y_pred, y)
#     if t % 100 == 99:
#         print(t, loss.item())
#
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#
#     # 更新权重
#     optimizer.step()
#
# print(list(model.parameters()))


class Animal:
    def __init__(self, type_name: str):
        self.type = type_name

    def __repr__(self):
        return "cc大傻瓜"

    def __str__(self):
        return "cc大傻瓜"

    def _abc(self): ...

    def abc(self):
        print("zyh")
        return "cyq"


a: Animal = Animal("Dog")
print(a)
