# coding=utf-8
import requests
import os


def get_img_data(word, pages):                      # PIG 输入关键词和页数
    '''
    PIG 从百度图片获取数据 GET
    '''
    params = []
    urls = []
    url = 'http://image.baidu.com/search/acjson'
    for i in range(30, 30*pages+30, 30):
        params.append({
                      'tn': 'resultjson_com',
                      'ipn': 'rj',
                      'ct': 201326592,
                      'is': '',
                      'fp': 'result',
                      'queryWord': word,
                      'cl': 2,
                      'lm': -1,
                      'ie': 'utf-8',
                      'oe': 'utf-8',
                      'adpicid': '',
                      'st': -1,
                      'z': '',
                      'ic': 0,
                      'word': word,
                      's': '',
                      'se': '',
                      'tab': '',
                      'width': '',
                      'height': '',
                      'face': 0,
                      'istype': 2,
                      'qc': '',
                      'nc': 1,
                      'fr': '',
                      'pn': i,
                      'rn': 30,
                      'gsm': '1e',
                      '1488942260214': ''
                      })

    for index, param in enumerate(params):
        print('params [%d/%d]' % (index, len(params)))
        response = requests.get(url, params=param)
        # print('response', response)
        # print('url', response.url)
        # print('request', response.request)
        print('text', response.text)

        print('response.status_code', response.status_code)
        if response.status_code == requests.codes.ok:
            # print('json()', response.json())
            data = response.json().get('data')
            print('data', data)
            print('data', len(data))
            del data[-1]                           # PIG 最后一个数据是空的所以删除
            urls += data
    return urls


def get_img_url(urls):
    '''
    PIG 从获得的数据中取得图片的地址 LET
    '''
    imgUrls = []
    for i in urls:
        imgUrl = i.get('objURL')
        imgUrls.append(imgUrl)
    return imgUrls


def parse_img_url(urls):
    '''
    PIG 解析百度图片的objURL地址 LET
    '''
    urlInfo = []
    for i in urls:
        i = str(i)
        i = i.replace('_z2C$q',':')
        i = i.replace('_z&e3B','.')
        i = i.replace('AzdH3F','/')
        intab = 'wkv1ju2it3hs4g5rq6fp7eo8dn9cm0bla'
        outtab = 'abcdefghijklmnopqrstuvw1234567890'
        trantab = str.maketrans(intab, outtab)
        i = i.translate(trantab)
        urlInfo.append(i)
    return urlInfo


def save_img(urls, path):           # PIG 填入图片路径
    '''
    PIG 存储图片 LET
    '''
    myHeaders = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'}
    if not os.path.exists(path):
        os.makedirs(path)
    for i,val in enumerate(urls):
        print('正在下载第{}张图片:'.format(i))
        try:
            pic = requests.get(val, headers=myHeaders, timeout=3).content
        except:
            print('失败')
        else:
            with open(path + word + '_' + str(i) + '.jpg', 'wb') as f:
                f.write(pic)
            print('成功')
    print("图片下载完成")


if __name__ == '__main__':
    keyword = list()
    for line in open('./keyword.txt', 'r'):
        line = line.rstrip().replace("\n", "")
        keyword.append(line)

    for word in keyword:
        pages = 30                  # PIG 输入页数，每页30张图片
        path = './car_data/'+word+'/'
        imgData = get_img_data(word, pages)
        imgUrl = get_img_url(imgData)
        imgUrl = parse_img_url(imgUrl)
        save_img(imgUrl, path)
