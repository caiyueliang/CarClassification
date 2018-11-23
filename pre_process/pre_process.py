# encoding:utf-8
import os
import shutil


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


#
def data_pre_process(root_path, output_path):
    files_list = list()

    for root, dirs, files in os.walk(root_path):
        for file in files:
            # print(file)     #文件名
            files_list.append(os.path.join(root, file))

    mkdir_if_not_exist(output_path)

    for file in files_list:
        print(file)
        name_list = file.split('/')[-1].split('_')

        new_dir = (name_list[0] + "_" + name_list[1]).replace(' ', '').replace('（', '(').replace('）', ')') \
            .replace("(stopsale)", "")

        mkdir_if_not_exist(os.path.join(output_path, new_dir))
        shutil.copy(file, os.path.join(output_path, new_dir, name_list[2]))
    return


def data_pre_process_1(root_path, output_path, min_size):
    mkdir_if_not_exist(output_path)
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            print(dir_path)              # 文件名

            for root_1, dirs_1, files_1 in os.walk(dir_path):
                if len(files_1) >= min_size:
                    shutil.copytree(dir_path, os.path.join(output_path, dir))
    return


def data_pre_process_2(root_path, output_path, count):
    mkdir_if_not_exist(output_path)

    mkdir_if_not_exist(output_path)
    mkdir_if_not_exist(os.path.join(output_path, 'train'))
    mkdir_if_not_exist(os.path.join(output_path, 'test'))

    for root, dirs, files in os.walk(root_path):
        for i, file in enumerate(files):
            dir_path = root.split('/')[-1]
            mkdir_if_not_exist(os.path.join(output_path, 'train', dir_path))
            mkdir_if_not_exist(os.path.join(output_path, 'test', dir_path))

            file_path = os.path.join(root, file)
            print(file_path)
            if i < count:
                shutil.copy(file_path, os.path.join(output_path, 'test', dir_path, file))
            else:
                shutil.copy(file_path, os.path.join(output_path, 'train', dir_path, file))
    return


if __name__ == '__main__':
    # data_pre_process('../../Data/car_classifier', '../../Data/car_classifier_new')
    # data_pre_process_1('../../Data/car_classifier_new/', '../../Data/car_classifier_min_50/', 50)
    data_pre_process_2('../../Data/car_classifier_min_50/', '../../Data/car_classifier_train/', 10)
