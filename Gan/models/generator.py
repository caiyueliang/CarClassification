# coding=utf-8
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        ngf = opt.ngf       # 生成器feature map数

        self.main = nn.Sequential(
            # in_channels (int): Number of channels in the input image
            # out_channels (int): Number of channels produced by the convolution
            # kernel_size (int or tuple): Size of the convolving kernel
            # stride (int or tuple, optional): Stride of the convolution. Default: 1
            # padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            #     will be added to both sides of each dimension in the input. Default: 0
            # output_padding (int or tuple, optional): Additional size added to one side
            #     of each dimension in the output shape. Default: 0
            # groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            # bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            # dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # shape: (ngf * 8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # shape: (ngf * 4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # shape: (ngf * 2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # shape: (ngf, 32, 32)

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # shape: (3, 96, 96)
        )

    def forward(self, input):
        return self.main(input)
