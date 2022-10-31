import requests
import json
import numpy as np
import utm
import time
import pickle
import pinyin

if __name__=='__main__':
    ak = "EB9xLNcPLkHWvxqu7ccZXER9rKnDrYqz"
    city_names = [
        '上海市',
        '北京市',
        '广州市',
        '深圳市',
        '成都市',
        '杭州市',
        '重庆市',
        '武汉市',
        '西安市',
        '苏州市',
        '天津市',
        '南京市',
        '长沙市',
        '郑州市',
        '东莞市',
        '青岛市',
        '合肥市',
        '宁波市',
        '佛山市',
        '昆明市',
        '沈阳市',
        '济南市',
        '无锡市',
        '厦门市',
        '福州市',
        '温州市',
        '金华市',
        '哈尔滨市',
        '南宁市',
        '大连市',
        '泉州市',
        '石家庄市',
        '贵阳市',
        '南昌市',
        '长春市',
        '惠州市',
        '常州市',
        '嘉兴市',
        '徐州市',
        '南通市',
        '太原市',
        '保定市',
        '珠海市',
        '中山市',
        '临沂市',
        '兰州市',
        '绍兴市',
        '潍坊市',
        '烟台'
    ]
    city_names_en = []
    lat_long = []
    utm_list = []
    for name in city_names:
        name_en = pinyin.get(name[:-1], format="strip", delimiter="")
        city_names_en.append(name_en)
        print(name_en)
        url = f"https://api.map.baidu.com/geocoding/v3/?address={name}&output=json&ak={ak}"
        req = requests.get(url)
        js = json.loads(req.text)
        print(f"name: {name} res:{req.text}")
        long = float(js['result']['location']['lng'])
        lat = float(js['result']['location']['lat'])
        lat_long.append((lat, long))
        east, north, zone_num, zone_letter = utm.from_latlon(lat, long, 51, 'R')
        print((east, north, zone_num, zone_letter))
        utm_list.append([east, north])
        time.sleep(0.1)

    lat_long = np.array(lat_long)
    utm_list = np.array(utm_list)

    data = {'lat_long': lat_long, 'utm_list': utm_list, 'cities': city_names, 'city_names_en': city_names_en}

    filehandler = open('data/geodata', 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()