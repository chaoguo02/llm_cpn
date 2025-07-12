import numpy as np
from deap import gp


def is_number(string):#创建一个函数 is_number，用于检查一个字符串是否可以转换为浮点数。如果可以，则返回 True，否则返回 False。
    try:
        float(string)
        return True
    except ValueError:
        return False

def protect_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def count_leaf_types(tree):
    leaf_count={}
    name_mapping = {
        'ARG0': 'a', 'ARG1': 'b', 'ARG2': 'c',
        'ARG3': 'd', 'ARG4': 'e', 'ARG5': 'f'
    }
    """统计每种叶子节点的数量"""
    for x in tree:
        if isinstance(x, gp.Terminal):  # 如果是叶子节点
            leaf_name = x.name
            if leaf_name in name_mapping:
                leaf_name = name_mapping[leaf_name]
            leaf_count[leaf_name] = leaf_count.get(leaf_name, 0) + 1
    return leaf_count