# coding:utf-8
import os
from torchvision.utils import save_image
from visdom import Visdom
import numpy as np


class Visual:
    def __init__(self, path):
        self.path = path
        self.vis = Visdom()

    def save_img(self, img, sub_path):
        img = 0.5 * (img + 1)
        img = img.clamp(0, 1)
        img = img.view(-1, 1, 28, 28)
        save_image(img, os.path.join(self.path, sub_path))

    '''
    1.在pytorch环境下开启服务:
    python -m visdom.server
    2.浏览器输入http://localhost:8097
    '''
    def show_img(self, img, name):
        img = img.numpy()
        self.vis.image(img, env='images', opts=dict(title=name))
