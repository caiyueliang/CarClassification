# encoding:utf-8
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
# from torchvision.datasets import ImageFolder
from my_folder import ImageFolder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from models.generator import Generator
import time


class ModuleTrain:
    def __init__(self, opt, best_acc=0.6):
        self.opt = opt
        self.best_acc = best_acc                        # 正确率这个值，才会保存模型

        self.netd = Discriminator(self.opt)
        self.netg = Generator(self.opt)
        self.use_gpu = False

        # 加载模型
        if os.path.exists(self.opt.netd_path):
            self.load(self.opt.netd_path)
        else:
            print('[Load model] error: %s not exist !!!' % self.opt.netd_path)
        if os.path.exists(self.opt.netg_path):
            self.load(self.opt.netg_path)
        else:
            print('[Load model] error: %s not exist !!!' % self.opt.netg_path)

        # DataLoader初始化
        self.transform_train = T.Compose([
            T.Resize((self.opt.img_size, self.opt.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])
        train_dataset = ImageFolder(root=self.opt.data_path, transform=self.transform_train, train=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                       num_workers=self.opt.num_workers, drop_last=True)

        # 优化器和损失函数
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr1, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr2, betas=(self.opt.beta1, 0.999))
        self.criterion = torch.nn.BCELoss()

        self.true_labels = Variable(torch.ones(self.opt.batch_size))
        self.fake_labels = Variable(torch.zeros(self.opt.batch_size))
        self.fix_noises = Variable(torch.randn(self.opt.batch_size, self.opt.nz, 1, 1))
        self.noises = Variable(torch.randn(self.opt.batch_size, self.opt.nz, 1, 1))

        # gpu or cpu
        if self.opt.use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        if self.use_gpu:
            print('[use gpu] ...')
            self.netd.cuda()
            self.netg.cuda()
            self.criterion.cuda()
            self.true_labels = self.true_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.fix_noises = self.fix_noises.cuda()
            self.noises = self.noises.cuda()
        else:
            print('[use cpu] ...')

        pass

    def train(self, epoch, decay_epoch=40, save_best=True):
        print('[train] epoch: %d' % epoch)
        for epoch_i in range(epoch):
            train_loss = 0.0
            correct = 0

            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
                self.lr = self.lr * 0.1
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            print('================================================')
            for batch_idx, (data, target) in enumerate(self.train_loader):              # 训练
                data, target = Variable(data), Variable(target)

                if self.use_gpu:
                    data = data.cuda()
                    target = target.cuda()

                # 梯度清0
                self.optimizer.zero_grad()
                # 计算损失
                output = self.model(data)
                loss = self.loss(output, target)
                # 反向传播计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()

                predict = torch.argmax(output, 1)
                correct += (predict == target).sum().data
                train_loss += loss.item()

            # print(correct)
            # print(len(self.train_loader.dataset))
            train_loss /= len(self.train_loader)
            acc = float(correct) / float(len(self.train_loader.dataset))
            print('[Train] Epoch: {} \tLoss: {:.6f}\tAcc: {:.6f}\tlr: {}'.format(epoch_i, train_loss, acc, self.lr))

            test_acc = self.test()
            if save_best is True:
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    str_list = self.model_file.split('.')
                    best_model_file = ""
                    for str_index in range(len(str_list)):
                        best_model_file = best_model_file + str_list[str_index]
                        if str_index == (len(str_list) - 2):
                            best_model_file += '_best'
                        if str_index != (len(str_list) - 1):
                            best_model_file += '.'
                    self.save(best_model_file)                                  # 保存最好的模型

        self.save(self.model_file)

    def test(self):
        test_loss = 0.0
        correct = 0

        time_start = time.time()
        # 测试集
        for data, target in self.test_loader:
            data, target = Variable(data), Variable(target)

            if self.use_gpu:
                data = data.cuda()
                target = target.cuda()

            output = self.model(data)
            # sum up batch loss
            if self.use_gpu:
                loss = self.loss(output, target)
            else:
                loss = self.loss(output, target)
            test_loss += loss.item()

            predict = torch.argmax(output, 1)
            correct += (predict == target).sum().data

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))
        test_loss /= len(self.test_loader)
        acc = float(correct) / float(len(self.test_loader.dataset))

        print('[Test] set: Test loss: {:.6f}\t Acc: {:.6f}\t time: {:.6f} \n'.format(test_loss, acc, time_avg))
        return acc

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)

