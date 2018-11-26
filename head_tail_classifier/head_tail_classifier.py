# coding=utf-8
from argparse import ArgumentParser
import os
import model_train
from torchvision import models


def parse_argvs():
    parser = ArgumentParser(description='head_tail_classifier')
    parser.add_argument('--train_path', type=str, help='train path', default='../../Data/head_tail_classifier/train')
    parser.add_argument('--test_path', type=str, help='test path', default='../../Data/head_tail_classifier/test')

    # parser.add_argument("--output_model_path", type=str, help="model path", default='./model/htc_squeezenet.pkl')
    parser.add_argument("--output_model_path", type=str, help="model path", default='./model/htc_resnet18.pkl')
    parser.add_argument('--classes_num', type=int, help='classes num', default=2)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--img_size', type=int, help='imgsize', default=224)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)

    input_args = parser.parse_args()
    print(input_args)
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    train_path = args.train_path
    test_path = args.test_path

    output_model_path = args.output_model_path
    num_classes = args.classes_num
    batch_size = args.batch_size
    img_size = args.img_size
    lr = args.lr

    print('train_path: %s' % train_path)
    print('test_path: %s' % test_path)
    print('output_model_path: %s' % output_model_path)
    print('num_classes: %d' % num_classes)
    print('img_size: %d' % img_size)
    print('batch_size: %d' % batch_size)
    print('lr: %s' % lr)

    model = models.resnet18(num_classes=num_classes)
    model = models.squeezenet1_1(num_classes=num_classes)
    model_train = model_train.ModuleTrain(train_path, test_path, output_model_path, model=model, batch_size=batch_size,
                                          img_size=img_size, lr=lr)

    model_train.train(200, 80)
    # model_train.test(show_img=True)
