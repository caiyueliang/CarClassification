# coding=utf-8
from argparse import ArgumentParser
import os
import model_train
from torchvision import models


def parse_argvs():
    parser = ArgumentParser(description='GAN')
    parser.add_argument('--data_path', type=str, help='data_path', default='./data')

    parser.add_argument('--num_workers', type=int, help='num_workers', default=1)
    parser.add_argument('--image_size', type=int, help='image_size', default=96)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
    parser.add_argument('--max_epoch', type=int, help='max_epoch', default=200)
    parser.add_argument('--lr1', type=float, help='learning rate 1', default=2e-4)
    parser.add_argument('--lr2', type=float, help='learning rate 2', default=2e-4)
    parser.add_argument('--beta1', type=float, help='Adam beta1', default=0.5)

    parser.add_argument('--use_gpu', type=bool, help='use_gpu', default=True)
    parser.add_argument('--nz', type=int, help='noise_channel', default=100)
    parser.add_argument('--ngf', type=int, help='生成器feature map', default=64)
    parser.add_argument('--ndf', type=int, help='判别器feature map', default=64)

    parser.add_argument('--save_path', type=str, help='image save path', default='./images')

    parser.add_argument('--vis', type=bool, help='visdom可视化', default=True)
    parser.add_argument("--env", type=str, help="visdom的env", default='GAN')
    parser.add_argument('--plot_every', type=int, help='间隔20画一次', default=20)

    parser.add_argument('--debug_file', type=str, help='存在该文件夹、进入debug模式', default='./tmp/debug_gan')
    parser.add_argument('--d_every', type=int, help='每1个batch训练一次判别器', default=1)
    parser.add_argument('--g_every', type=int, help='每5个batch训练一次生成器', default=5)
    parser.add_argument('--decay_every', type=int, help='每10个epoch保存一次模型', default=10)
    parser.add_argument('--netd_path', type=str, help='model_path', default='./checkpoints/netd.pth')
    parser.add_argument('--netg_path', type=str, help='model_path', default='./checkpoints/netg.pth')

    parser.add_argument('--gen_img', type=str, help='gen_img', default='result.png')
    parser.add_argument('--gen_num', type=int, help='从512张生成的图片中保存最好的64张', default=64)
    parser.add_argument('--gen_search_num', type=int, help='gen_search_num', default=512)
    parser.add_argument('--gen_mean', type=int, help='噪声均值', default=0)
    parser.add_argument('--gen_std', type=int, help='噪声方差', default=1)

    input_args = parser.parse_args()
    print(input_args)
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    # train_path = args.train_path
    # test_path = args.test_path
    # output_model_path = args.output_model_path
    # num_classes = args.classes_num
    # batch_size = args.batch_size
    # img_size = args.img_size
    # lr = args.lr

    # model = models.resnet18(num_classes=num_classes)
    # model = models.squeezenet1_1(num_classes=num_classes)
    model_train = model_train.ModuleTrain(opt=args)

    model_train.train()
    # model_train.test(show_img=True)
