# coding=utf-8
from argparse import ArgumentParser
import os
import model_train
from torchvision import models
import cv2
import torch
import shutil
from torchvision import transforms as T
from torch.autograd import Variable
from PIL import Image


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


class HeadTailPredict:
    def __init__(self, model, model_file, img_size):
        self.model = model
        self.model_file = model_file
        self.img_size = img_size

        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False

        if self.use_gpu:
            print('[use gpu] ...')
            self.model = self.model.cuda()

        # 加载模型
        if os.path.exists(self.model_file):
            self.load(self.model_file)
        else:
            print('[ERROR] model file not exists ...')

        self.transform_test = T.Compose([
            # T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def predict(self, im):
        image = cv2.resize(im, (self.img_size, self.img_size))
        img_tensor = self.transform_test(image)
        img_tensor = img_tensor.view(-1, 3, self.img_size, self.img_size)
        img_var = Variable(img_tensor)
        if self.use_gpu:
            img_var = img_var.cuda()

        output = self.model(img_var)
        predict = torch.argmax(output, 1)
        return predict.item()

    def predict_image(self, root_path, output_path):
        files_list = list()

        for root, dirs, files in os.walk(root_path):
            for file in files:
                # print(file)     #文件名
                files_list.append(os.path.join(root, file))

        head_path = os.path.join(output_path, 'head')
        tail_path = os.path.join(output_path, 'tail')
        mkdir_if_not_exist(output_path)
        mkdir_if_not_exist(head_path)
        mkdir_if_not_exist(tail_path)

        for file in files_list:
            print(file)
            image = cv2.imread(file)
            # cv2.imshow('image', image)
            label = self.predict(image)

            str_list = file.split('/')
            save_image_name = str_list[-2] + '_' + str_list[-1]
            if label == 0:
                shutil.copy(file, os.path.join(head_path, save_image_name))
            elif label == 1:
                shutil.copy(file, os.path.join(tail_path, save_image_name))
            # cv2.waitKey(0)

            # shutil.copy(file, os.path.join(output_path, new_dir, name_list[2]))
        return

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)


def parse_argvs():
    parser = ArgumentParser(description='head_tail_classifier')
    # parser.add_argument("--model_path", type=str, help="model path", default='./model/htc_squeezenet.pkl')
    parser.add_argument("--model_path", type=str, help="model path", default='./checkpoints/htc_resnet18_best.pkl')
    parser.add_argument('--classes_num', type=int, help='classes num', default=2)
    parser.add_argument('--img_size', type=int, help='imgsize', default=224)

    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    model_path = args.model_path
    num_classes = args.classes_num
    img_size = args.img_size

    print('model_path: %s' % model_path)
    print('num_classes: %d' % num_classes)
    print('img_size: %d' % img_size)

    model = models.resnet18(num_classes=num_classes)
    # model = models.squeezenet1_1(num_classes=num_classes)
    predict = HeadTailPredict(model=model, model_file=model_path, img_size=img_size)

    predict.predict_image('../../Data/car_classifier_new/', '../../Data/car_head_classifier/')
    # model_train.test(show_img=True)
