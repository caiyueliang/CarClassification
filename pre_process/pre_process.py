# encoding:utf-8
import os
import shutil


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def data_pre_process(root_path, input_path):
    files_list = list()

    for root, dirs, files in os.walk(root_path):
        for file in files:
            # print(file)     #文件名
            files_list.append(os.path.join(root, file))

    mkdir_if_not_exist(input_path)

    for file in files_list:
        print(file)
        name_list = file.split('/')[-1].split('_')

        new_dir = (name_list[0] + "_" + name_list[1]).replace(' ', '').replace('（', '(').replace('）', ')') \
            .replace("(stopsale)", "")

        mkdir_if_not_exist(os.path.join(input_path, new_dir))
        shutil.copy(file, os.path.join(input_path, new_dir, name_list[2]))

    print len(files_list)
    return


if __name__ == '__main__':
    data_pre_process('../../Data/car_classifier', '../../Data/car_classifier_new')