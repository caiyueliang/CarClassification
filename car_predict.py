# coding=utf-8
from argparse import ArgumentParser
import os
from torchvision import models
import cv2
import torch
import shutil
from torchvision import transforms as T
from torch.autograd import Variable

car_label = ["GMC_SAVANA", "Jeep(进口)_大切诺基(进口)", "Jeep(进口)_牧马人", "Jeep(进口)_自由侠(海外)", "MINI_MINICOUNTRYMAN", "smart_smartforfour", "smart_smartfortwo", "一汽-大众_宝来", "一汽丰田_普拉多", "一汽吉林_森雅R7|2017", "三菱(进口)_三菱i", "三菱(进口)_帕杰罗(进口)", "上汽大众_朗逸", "上汽大众_桑塔纳", "上汽大通_上汽大通G10", "上汽大通_上汽大通V80", "上汽斯柯达_明锐", "上汽通用五菱_五菱之光", "上汽通用五菱_五菱荣光", "上汽通用五菱_宝骏730", "上汽通用五菱_宝骏E100", "上汽集团_名爵3", "东南汽车_东南DX7", "东风乘用车_东风风神AX7", "东风小康_东风风光370", "东风日产_骊威", "东风本田_本田CR-V", "东风标致_标致206", "东风风行_景逸X5", "东风风行_菱智", "东风风行_风行CM7", "东风风行_风行SX6", "众泰汽车_大迈X5", "克莱斯勒(进口)_克莱斯勒300C(进口)", "兰博基尼_Aventador", "劳斯莱斯_幻影", "北京汽车_北京BJ40", "北京汽车制造厂_战旗", "北京现代_北京现代ix35", "北汽银翔_北汽幻速S3", "双环汽车_双环SCEO", "双环汽车_小贵族", "双龙汽车_爱腾", "大众(进口)_凯路威", "大众(进口)_夏朗", "大众(进口)_迈特威", "大众(进口)_途锐", "奇瑞汽车_奇瑞QQ3", "奇瑞汽车_瑞虎3", "奔驰(进口)_奔驰B级", "奔驰(进口)_奔驰GLE", "奔驰(进口)_奔驰G级", "宾利_慕尚", "广汽丰田_汉兰达", "广汽乘用车_传祺GS4", "斯柯达(进口)_Yeti(进口)", "日产(进口)_Pathfinder", "昌河铃木_北斗星", "林肯_林肯MKZ", "林肯_领航员", "欧宝_欧宝Corsa", "江淮汽车_瑞风S3", "江铃汽车_驭胜S350", "沃尔沃(进口)_沃尔沃V40", "猎豹汽车_猎豹CS10", "现代(进口)_现代i10", "福特(进口)_嘉年华(进口)", "福特(进口)_福克斯(进口)", "福特(进口)_福特F-150", "绵阳金杯_智尚S30", "英菲尼迪(进口)_英菲尼迪Q70", "菲亚特(进口)_Panda", "菲亚特(进口)_朋多", "菲亚特(进口)_菲亚特500", "讴歌(进口)_讴歌RLX", "起亚(进口)_Picanto", "起亚(进口)_Soul", "路虎(进口)_发现", "道奇(进口)_道奇Ram", "道奇(进口)_酷威", "野马汽车_野马T70", "铃木(进口)_吉姆尼(进口)", "长城汽车_哈弗H1", "长城汽车_哈弗H5", "长城汽车_哈弗H6Coupe", "长城汽车_哈弗H8", "长安汽车_奔奔i", "长安汽车_长安CS35", "长安汽车_长安CS75", "长安福特_翼搏", "长安福特_翼虎", "长安福特_锐界", "长安铃木_奥拓", "长安马自达_马自达2", "长安马自达_马自达3Axela昂克赛拉", "长安马自达_马自达CX-5", "阿尔法·罗密欧_Giulietta", "雪佛兰(进口)_斯帕可", "雪铁龙(进口)_DS3经典", "雷诺(进口)_Twingo"]


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


class CarPredict:
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
        # print(train_dataset.classes)

        return predict.item()

    def predict_image(self, root_path):
        files_list = list()

        for root, dirs, files in os.walk(root_path):
            for file in files:
                files_list.append(os.path.join(root, file))

        for file in files_list:
            print(file)
            image = cv2.imread(file)
            cv2.imshow('image', image)
            label = self.predict(image)
            print(label)
            print(car_label[label])
            cv2.waitKey(0)
        return

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))


def parse_argvs():
    parser = ArgumentParser(description='car_classifier')
    parser.add_argument("--model_path", type=str, help="model path", default='./checkpoints/cc_resnet18_best.pkl')
    parser.add_argument('--classes_num', type=int, help='classes num', default=100)
    parser.add_argument('--img_size', type=int, help='img size', default=224)

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
    predict = CarPredict(model=model, model_file=model_path, img_size=img_size)

    predict.predict_image('../../Data/car_classifier/vanke_car/data_clean/')

