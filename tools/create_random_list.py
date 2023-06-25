'''
Description: 生成模型库中的list.txt文件
Author: wangdx
Date: 2021-11-10 15:10:24
LastEditTime: 2021-12-20 11:30:36
'''

import os
import glob
import random

path = 'E:/research/dataset/grasp/sim_grasp/mesh_database/dataset/test'

files = glob.glob(os.path.join(path, '*.urdf'))
random.shuffle(files)

txt = open(path + './../list_test.txt', 'w+')
for f in files:
    fname = os.path.basename(f)
    pre_fname = os.path.basename(os.path.dirname(f))
    txt.write(pre_fname + '/' + fname[:-5] + '\n')
txt.close()
print('done')

    
