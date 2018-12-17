# coding=utf-8
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ngf = opt.ngf       # 生成器feature map数

        self.main = nn.Sequential(
            # input: (3, 96, 96)
            nn.Conv2d(3, ngf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (ngf, 32, 32)

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (ngf * 2, 16, 16)

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (ngf * 4, 8, 8)

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (ngf * 8, 4, 4)

            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()    # 输出一个数（概率）
        )

    def forward(self, input):
        return self.main(input).view(-1)
