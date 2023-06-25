'''
Description: 
Author: wangdx
Date: 2021-01-20 20:22:48
LastEditTime: 2021-09-25 09:59:34

创建txt文件，存储数据集样本文件列表
每个类别(dex/egad/clutter)内两个txt文件，分别存储test和train文件
'''

import os
import glob


neighbor_val = 0.8


def writeTxt(paths, txtName):
    """
    把paths中的路径的文件名写入txtName中，一行一个元素

    paths: 路径list
    txtName: *.txt
    """
    f = open(txtName, 'w')
    # 抓取点置信度为1
    for path in paths:
        filename = os.path.basename(path)
        f.write(str(filename) + '\n')
    f.close()


def run():
    dataset_path = 'F:/my_dense/dataset/regression'      # 数据集路径

    # 搜索dataset_path下的文件夹
    paths = os.listdir(dataset_path)
    for path in paths:
        path = os.path.join(dataset_path, path)
        if not os.path.isdir(path): # 过滤文件
            continue
        test_path = os.path.join(path, 'test')
        train_path = os.path.join(path, 'train')
        if not os.path.exists(test_path) or not os.path.exists(train_path):
            continue

        print('writing...', path)
        
        # 获取depth文件名
        test_files = glob.glob(os.path.join(test_path, '*_depth.mat'))
        train_files = glob.glob(os.path.join(train_path, '*_depth.mat'))
        # 写入txt文件
        test_txtName = os.path.join(path, 'test.txt')
        train_txtName = os.path.join(path, 'train.txt')
        writeTxt(test_files, test_txtName)
        writeTxt(train_files, train_txtName)
        

        

    

    



if __name__ == "__main__":
    run()
