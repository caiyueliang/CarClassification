# coding=utf-8
from argparse import ArgumentParser
import os
import model_train
import torch
from torchvision import models
from models import resnet_torch


def parse_argvs():
    parser = ArgumentParser(description='car_classifier')
    parser.add_argument('--train_path', type=str, help='train dataset path',
                        default='../Data/car_classifier/classifier_train_best/train')
    parser.add_argument('--test_path', type=str, help='test dataset path',
                        default='../Data/car_classifier/classifier_train_best/test')

    parser.add_argument("--model_name", type=str, help="model name", default='densenet121')
    parser.add_argument("--output_model_path", type=str, help="output model path", default='./checkpoints')
    parser.add_argument('--classes_num', type=int, help='classes num', default=4)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--img_size', type=int, help='img size', default=224)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument("--optimizer", type=str, help="optimizer", default='Adam')
    parser.add_argument("--re_train", type=bool, help="re_train", default=False)

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

    if args.model_name == 'resnet34':
        model_file = os.path.join(output_model_path, 'cc_resnet34_' + str(args.classes_num) + '.pkl')
        # model = models.resnet34(num_classes=num_classes)
        model = resnet_torch.resnet34(num_classes=num_classes)
    elif args.model_name == 'densenet121':
        model_file = os.path.join(output_model_path, 'cc_densenet121_' + str(args.classes_num) + '.pkl')
        model = models.densenet121(num_classes=num_classes)
    elif args.model_name == 'inception_v3':
        img_size = 299
        model_file = os.path.join(output_model_path, 'cc_inception_v3_' + str(args.classes_num) + '.pkl')
        model = models.inception_v3(num_classes=num_classes, aux_logits=False)
    else:
        model_file = os.path.join(output_model_path, 'cc_resnet18_' + str(args.classes_num) + '.pkl')
        # model = models.resnet18(pretrained=True)
        # fc_features = model.fc.in_features                              # 提取fc层中固定的输入参数
        # model.fc = torch.nn.Linear(fc_features, num_classes)            # 修改类别为num_classes
        model = resnet_torch.resnet18(num_classes=num_classes)

    model_train = model_train.ModuleTrain(train_path=train_path, test_path=test_path, model_file=model_file,
                                          model=model, batch_size=batch_size, img_size=img_size, lr=lr,
                                          optimizer=args.optimizer, re_train=args.re_train)

    model_train.train(200, 100)
