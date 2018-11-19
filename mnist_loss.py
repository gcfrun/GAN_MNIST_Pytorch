# coding:utf-8
import torch.nn as nn


# 损失函数
class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
