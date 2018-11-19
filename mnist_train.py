# coding:utf-8
from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from mnist_data import Mnist
from mnist_loss import Loss
from mnist_net import Discriminator, Generator
from mnist_visual import Visual


def main(args):
    # 1.相关路径
    # 模型存储路径
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # 数据集路径
    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)
    # 可视化路径
    if not os.path.exists(args.visualdir):
        os.makedirs(args.visualdir)

    # 2.数据加载
    dataset_train = Mnist(args.datadir).train_data()
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 3.初始化模型
    D = Discriminator()
    G = Generator(args.z_dimension, 3136)
    if args.cuda:
        D = torch.nn.DataParallel(D).cuda()
        G = torch.nn.DataParallel(G).cuda()

    # 4.优化器
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

    # 5.损失函数
    criterion = Loss()

    # 6.可视化
    visual = Visual(args.visualdir)

    # 7.恢复模型
    start_epoch = 0
    if args.resume:
        d_path = args.savedir + '/discriminator.pth'
        assert os.path.exists(
            d_path), "Error: resume option was used but discriminator.pth was not found in folder"
        d_checkpoint = torch.load(d_path)
        start_epoch = d_checkpoint['epoch']
        D.load_state_dict(d_checkpoint['state_dict'])

        g_path = args.savedir + '/generator.pth'
        assert os.path.exists(
            g_path), "Error: resume option was used but generator.pth was not found in folder"
        g_checkpoint = torch.load(g_path)
        G.load_state_dict(g_checkpoint)

        print("=> Loaded checkpoint at epoch {})".format(start_epoch))

    # 8.开始训练
    print("========== TRAINING  START===========")
    for epoch in range(start_epoch + 1, args.num_epochs):
        for i, (img, _) in enumerate(loader_train):
            num_img = img.size(0)
            # =================数据处理
            # 真实图片
            real_img = Variable(img)
            # 真样本1
            real_label = Variable(torch.ones(num_img))
            # 假样本0
            fake_label = Variable(torch.zeros(num_img))
            # 用于判别器的噪声
            d_z = Variable(torch.randn(num_img, args.z_dimension))
            # 用于生成器的噪声
            g_z = Variable(torch.randn(num_img, args.z_dimension))
            if args.cuda:
                real_img = real_img.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                d_z = d_z.cuda()
                g_z = g_z.cuda()

            # =================训练判别器
            # 真实图片loss
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better

            # 假图片loss
            fake_img = G(d_z)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better

            # 判别器梯度反传和参数优化
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===============训练生成器
            # 假图片loss
            fake_img = G(g_z)
            output = D(fake_img)
            g_loss = criterion(output, real_label)

            # 生成器梯度反传和参数优化
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # =================打印
            if (i + 1) % args.steps_loss == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                      'D real: {:.6f}, D fake: {:.6f}'
                      .format(epoch, args.num_epochs, d_loss.data[0], g_loss.data[0],
                              real_scores.data.mean(), fake_scores.data.mean()))

        # =================可视化
        if epoch == 1:
            visual.save_img(real_img.cpu().data, 'real_images.png')
            visual.show_img(real_img[0].cpu().data, 'real_image')
        if epoch % args.epochs_visual == 0:
            visual.save_img(fake_img.cpu().data, 'fake_images-{}.png'.format(epoch))
            visual.show_img(fake_img[0].cpu().data, 'fake_image (epoch: %d)' % epoch)

        # =================保存模型
        if epoch % args.epochs_save == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': D.state_dict()
            }, args.savedir + '/discriminator.pth')
            torch.save(G.state_dict(), args.savedir + '/generator.pth')

    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # 指定数据集路径
    parser.add_argument('--datadir', default='./data')
    # 存储日志和模型的路径
    parser.add_argument('--savedir', default='./model')
    # 可视化保存图片路径
    parser.add_argument('--visualdir', default='./visual')

    # 打印loss间隔，单位step
    parser.add_argument('--steps-loss', type=int, default=100)
    # 可视化间隔，单位epoch
    parser.add_argument('--epochs-visual', type=int, default=1)
    # 存储模型间隔，单位epoch
    parser.add_argument('--epochs-save', type=int, default=1)

    # 训练的epoch数
    parser.add_argument('--num-epochs', type=int, default=100)
    # 线程数
    parser.add_argument('--num-workers', type=int, default=4)
    # 训练批大小
    parser.add_argument('--batch-size', type=int, default=128)
    # 输入噪声的维度
    parser.add_argument('--z-dimension', type=int, default=100)

    # 是否使用cuda
    parser.add_argument('--cuda', action='store_true', default=True)
    # 是否重新使用权重
    parser.add_argument('--resume', action='store_true')

    main(parser.parse_args())
