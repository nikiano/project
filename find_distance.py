import re
import math
from collections import defaultdict

coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""


# 获取信息
def get_massage(text):
    city_position = {}
    for line in text.split('\n'):
        if not line:
            continue
        name = re.findall(r"name:'(\w+)'", line)[0]
        position = re.findall(r"Coord:\[(\d+.\d+),\s(\d+.\d+)\]", line)[0]
        position = tuple(map(float, position))
        city_position[name] = position
    return city_position


# 计算距离：（弧度的距离）
def geo_distance(origin, destination):
    x1, y1 = origin
    x2, y2 = destination
    r = 6371
    d = r * math.acos(math.sin(math.radians(x1)) *
                      math.sin(math.radians(x2)) +
                      math.cos(math.radians(x1)) *
                      math.cos(math.radians(x2)) *
                      math.cos(math.radians(y1-y2)))
    return d


# 连接符合条件的点
def distance(city_info):
    city_name = city_info.keys()
    city_connection = defaultdict(list)
    for c1 in city_name:
        for c2 in city_name:
            if c1 == c2:
                continue
            if geo_distance(city_info[c1], city_info[c2]) < 700:
                city_connection[c1].append(c2)
    return city_connection


# 寻找两点之间的最短/最长距离
def search(graph, start, end, short):
    if short:
        short = 0
    else:
        short = -1
    path = [[start]]
    visted = set()

    while path:
        a = path.pop(short)
        b = a[-1]
        if b in visted:
            continue
        for i in graph[b]:
            if i in path:
                continue
            path.append(a+[i])
            if i == end:
                return print(a+[i])
        visted.add(b)


city = get_massage(coordination_source)
city_graph = distance(city)
search(city_graph, "上海", "香港", True)
