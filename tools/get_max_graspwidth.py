'''
Description:  获取数据集中最大的抓取宽度
Author: wangdx
Date: 2021-10-23 14:33:26
LastEditTime: 2021-10-23 14:39:27
'''

import sys
import os
import glob
sys.path.append(os.curdir)


def get_example_graspwidth(file):
    """
    获取该样本中最大的抓取宽度

    file: txt格式的抓取回归标签文件
    """
    max_graspwidth = 0.0
    # 如果txt文件的最后一行没有字符，则f.readline()会返回None
    mode = 0
    with open(file) as f:
        while 1:
            line = f.readline()
            if not line:
                break
            strs = line.split(' ')
            if len(strs) != 4:
                mode += 1
                continue
            
            max_graspwidth = max(max_graspwidth, float(strs[3]))

    return max_graspwidth


def get_max_graspwidth(dataset_path, start_id=0):
    """
    dataset_path: 数据集路径
    start_id: 从start_id个样本开始可视化
    """
    grasp_label_files = glob.glob(os.path.join(dataset_path, '*grasp.txt'))

    max_graspwidth = 0.0
    for grasp_label_file in grasp_label_files[start_id:]:
        # 读取txt格式的抓取
        grasp_width = get_example_graspwidth(grasp_label_file)
        max_graspwidth = max(max_graspwidth, grasp_width)

        print(grasp_label_file)
        print(max_graspwidth)
    


if __name__ == "__main__":
    get_max_graspwidth('F:/my_dense/dataset/regression/clutter/test')

    