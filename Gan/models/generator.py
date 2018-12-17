# coding=utf-8
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        ngf = opt.ngf       # 生成器feature map数

        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # shape: (ngf * 8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # shape: (ngf * 4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # shape: (ngf * 2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # shape: (ngf, 32, 32)

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # shape: (3, 96, 96)
        )

    def forward(self, input):
        return self.main(input)
