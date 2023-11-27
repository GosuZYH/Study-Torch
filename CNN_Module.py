#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved
#
# Time    : 2023/11/10 15:23
# Author  : GosuXX
# File    : CNN_Module.py

import torch.nn as nn
import torch


class CNN_Module(nn.Module):
    """ 卷积神经网络 """

    def __init__(self):
        super().__init__()
        print(f"是否支持GPU读取: {torch.cuda.is_available()}")
        '''Sequential为一个容器，按顺序形成模块'''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 20, 5),  # 卷积层1
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )
        ''' Dropout通过在训练时随机将部分神经元的输出置为0,
        从而减少神经元之间的耦合，减少过拟合的风险 
        args @p：对于输入中各个元素zero out的概率，也就是说当p=1时，输出为全0 
             @inplace：表示是否对tensor本身操作，若选择True,将会设置tensor为0'''
        self.drop_out = nn.Dropout()
        ''' Linear为创建全连接层（线性层）
        全连接层的作用是将输入与权重矩阵相乘并加上偏置，
        然后可以通过激活函数进行非线性变换
        args @in_features: 输入的神经个数
             @out_features：输出的神经数'''
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forTest(self):
        rct = torch.rand(5, 3)  # 生成随机5X3矩阵,M组数据样本,每个样本N个特征
        print(rct)
        model = nn.Linear(3,1)  # 接收N个特征 输出N个特征
        print(model(rct))

    def forward(self, x): ...



if __name__ == "__main__":
    cnn = CNN_Module()
    cnn.forTest()
