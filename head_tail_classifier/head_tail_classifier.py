# coding=utf-8
import os
import model_train
from torchvision import models

if __name__ == '__main__':
    train_path = '../Data/car_classifier_train/train'
    test_path = '../Data/car_classifier_train/test'

    FILE_PATH = './model/resnet18_params.pkl'
    num_classes = 356
    batch_size = 100
    img_size = 224
    lr = 1e-2

    print('train_path: %s' % train_path)
    print('test_path: %s' % test_path)
    print('num_classes: %d' % num_classes)
    print('img_size: %d' % img_size)
    print('batch_size: %d' % batch_size)

    model = models.resnet18(num_classes=num_classes)
    model_train = model_train.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=batch_size,
                                          img_size=img_size, lr=lr)

    model_train.train(200, 60)
    # model_train.test(show_img=True)
