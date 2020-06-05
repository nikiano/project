import requests
import re
from collections  import defaultdict
import math
"""
获取地铁信息，查找站点间的最短距离
"""

# 整理信息
def sub_info(r):
    b = r.text.split('st')[1:]
    lines_info = {}  # 线路：站点
    stations_info = {}  # 站点：位置
    for i in b:
        station = re.findall(r'"n":"(\w+)"', i)
        x_y = re.findall('"sl":"(\d+.\d+),(\d+.\d+)"', i)
        x_y = [tuple(map(float, i)) for i in x_y]
        sub = re.findall('"ln":"(\w+)"', i)[0]

        lines_info[sub] = station
        for n, d in zip(station, x_y):
            stations_info[n] = d
    return lines_info, stations_info


# 构建关系图
def get_neighbor_info(lines_info):
    neighbor_info = defaultdict(list)

    # 把str2加入str1站点的邻接表中
    def add_neighbor_dict(info, str1, str2):
        info[str1].append(str2)

    for line, stations in lines_info.items():
        sta_len = len(stations)
        for i in range(sta_len - 1):
            add_neighbor_dict(neighbor_info, stations[i], stations[i + 1])
            add_neighbor_dict(neighbor_info, stations[sta_len - 1 - i], stations[sta_len - 2 - i])

    return neighbor_info


# 以路径为cost的启发式搜索
def short_path(pathes):
    def get_distance(path):
        distance = 0
        for i, station in enumerate(path[:-1]):
            a = stations_info[station]
            b = stations_info[path[i + 1]]
            distance += math.hypot(a[0] - b[0], a[1] - b[1])
        return distance
    return sorted(pathes, key=get_distance)


# 获取路径
def get_path(neighbor_info, from_station, to_station, search_stratery=None):
    path = [[from_station]]
    visited = set()
    while path:
        a = path.pop(0)
        f = a[-1]
        if f in visited: continue
        for i in neighbor_info[f]:
            if i in a: continue
            path.append(a + [i])
            if not search_stratery:
                if i == to_station:
                    return path[-1]
        visited.add(f)
        if search_stratery:
            path = search_stratery(path)
            if path and path[0][-1] == to_station:
                return path[0]

# 搜索站点所在的线路
def path_line(lines_info, node):
    line_list = []
    for line, station in lines_info.items():
        if node in station:
            line_list.append(line)
    return line_list

# 打印文本信息
def get_path_DFS_ALL(lines_info, neighbor_info, from_station, to_station, search_stratery=None):
    # 途经节点
    path = get_path(neighbor_info, from_station, to_station, search_stratery)

    # 节点总包含的线路
    station_line_list = []
    for i in path:
        station_line_list.append(path_line(lines_info, i))

    # 是否需要换乘
    test = ""
    for i in range(len(station_line_list) - 1):
        if i == 0:
            for s in station_line_list[i]:
                if s in station_line_list[i + 1]:
                    a = s
                    test += f"{a},{path[0]}上车，"
        for s in station_line_list[i]:
            if s in station_line_list[i + 1]:
                if a != s:
                    a = s
                    test += f"{path[i]}下车， 换乘{s}，"
    test += f"{path[-1]}下车"
    return test


r = requests.get('http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json')
lines_info, stations_info = sub_info(r)
neighbor_info = get_neighbor_info(lines_info)

print(get_path_DFS_ALL(lines_info, neighbor_info,'宋家庄', '牡丹园', search_stratery=short_path))
