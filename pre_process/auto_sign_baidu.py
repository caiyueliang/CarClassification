# coding:utf-8
import os
import urllib, urllib2, base64
import sys
import ssl
import json
import shutil

my_access_token = ['', '']


def get_access_token():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=car&client_secret=111'
    request = urllib2.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib2.urlopen(request)
    content = response.read()
    if (content):
        print(content)


def post_image_base64_baidu(image_path, access_token):
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/car"

    # 二进制方式打开图片文件
    f = open(image_path, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img, "top_num": 5}
    params = urllib.urlencode(params)

    # access_token = '[调用鉴权接口获取的token]'
    request_url = request_url + "?access_token=" + access_token
    request = urllib2.Request(url=request_url, data=params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib2.urlopen(request)
    content = response.read()
    if content:
        print content
        decode_json = json.loads(content)
        print decode_json
        return True, False, decode_json
    if '' in content:
        # access_token异常，次数超过，更换token
        return False, True, None
    return False, False, None


# 自动标记
def auto_sign(root_path):
    use_token_id = 0

    for root, dirs, files in os.walk(root_path):
        for file in files:
            old_file_path = os.path.join(root, file)

            # 没有百度标志，表示没进行识别
            if 'baidu' not in file:
                print old_file_path, os.path.exists(old_file_path)
                success, token_err, result = post_image_base64_baidu(old_file_path, my_access_token[use_token_id])
                if success:
                    new_file_name = file.split('.')[0] + '_baidu_' + result['result'][0]['name'] + \
                                    '_' + str(result['result'][0]['score']) + file.split('.')[-1]
                    new_file_path = os.path.join(root, new_file_name)
                    shutil.move(old_file_path, new_file_path)
                else:
                    new_file_name = file.split('.')[0] + '_baidu_None_0.0_' + file.split('.')[-1]
                    new_file_path = os.path.join(root, new_file_name)
                    shutil.move(old_file_path, new_file_path)

                if token_err:
                    if use_token_id < len(my_access_token) - 1:
                        use_token_id += 1
                    else:
                        print('[token_err] all access_token used !')
                        return


if __name__ == '__main__':
    # post_image_base64('http://127.0.0.1:9511')
    # post_image_base64('http://lpr.maysatech.com')

    # 调用百度接口
    # get_access_token()
    # post_image_base64_baidu('./奥迪A3_9.jpg')

    # 自动标记
    auto_sign('../../Data/car_classifier/clean_car/car_data')
