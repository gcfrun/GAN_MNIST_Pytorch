# coding:utf-8
import torch.nn as nn


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义第一层网络
        self.conv1 = nn.Sequential(
            # 输入1通道，输出32通道，卷积核大小5，边缘填充2
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            # 负区域斜率为0.2
            nn.LeakyReLU(0.2, True),
            # 卷积核大小2，步长2
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        # 定义第二层网络
        self.conv2 = nn.Sequential(
            # 输入32通道，输出64通道，卷积核大小5，边缘填充2
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        # 共2个全连接层
        self.fc = nn.Sequential(
            # 输入向量长度64*7*×7，输出长度1024
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            # 输入向量长度1024，输出长度1
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        # reshape
        x = x.view(x.size(0), -1)  # batch,width*height,*channel
        x = self.fc(x)
        x = x.squeeze()     # 压缩尺寸保证与真值同一个shape
        return x


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        # 全连接
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            # 输入1通道，输出50通道，卷积核大小3，步长1，边缘填充1
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            # 输入通道50，输出通道25，卷积核大小3，步长1，边缘填充1
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            # 输入通道25，输出通道1，卷积核大小2，步长2，边缘填充1
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
