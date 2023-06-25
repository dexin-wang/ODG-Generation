'''
Description: 统计抓取标签的数量
Author: wangdx
Date: 2021-01-20 20:22:48
LastEditTime: 2022-01-17 13:59:05
'''

import os
import glob



def count_label(src_path):
    """
    获取标注的抓取数量
    """
    # src_path = 'F:/sim_grasp/dataset'        # 抓取源数据
    count = 0
    label_files = glob.glob(os.path.join(src_path, '*', '*.txt'))
    for file in label_files:
        with open(file) as f:
            while 1:
                line = f.readline()
                strs = line.split(' ')
                if len(strs) != 5:
                    break
                count += 1
    
        print(file)
        print(count)


def count_label_enhance(src_path):
    """
    获取数据增强后抓取的数量
    """
    # src_path = 'F:/sim_grasp/dataset'        # 抓取源数据
    count = 0
    label_files = glob.glob(os.path.join(src_path, '*.txt'))
    for file in label_files:
        with open(file) as f:
            count += len(f.readlines())
    
        print(file)
        print(count)



if __name__ == "__main__":
    path = 'E:/research/dataset/grasp/cornell/wdx_sgdn_new/data/clutter'
    # count_label(path)
    count_label_enhance(path)
    
    # 正负样本比例
    # c_p = 8876142
    # c_n = 3045 * 480 * 640 * 18 - c_p
    # print(c_n/c_p)