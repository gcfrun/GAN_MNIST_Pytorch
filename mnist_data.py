# coding:utf-8
from torchvision import transforms
from torchvision import datasets


class Mnist:
    def __init__(self, data_path):
        # 数据路径
        self.data_path = data_path
        # 数据预处理，当然预处理还有其他方式：翻转、平移、裁剪...
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 获取训练数据
    def train_data(self):
        return datasets.MNIST(self.data_path, train=True, transform=self.img_transform)

    # 获取测试数据
    def test_data(self):
        return datasets.MNIST(self.data_path, train=False, transform=self.img_transform)
