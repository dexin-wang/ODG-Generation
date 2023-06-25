'''
Description: 为现有的obj创建collision obj 模型
Author: wangdx
Date: 2021-11-20 21:53:02
LastEditTime: 2022-01-07 15:40:11
'''

import os
import pybullet as p


path = 'E:/research/dataset/grasp/ShapeNet/ShapeNetSem/test/1a4ec387ea6820345778775dfd5ca46a'

# 为文件夹内所有obj生成col
# p.connect(p.DIRECT)
# files = os.listdir(path)
# for file in files:
#     if not file.endswith('.obj'):
#         continue
#     print('processing ...', file)
#     name_in = os.path.join(path, file)
#     name_out = os.path.join(path, file.replace('.obj', '_col.obj'))
#     name_log = "log.txt"
#     p.vhacd(name_in, name_out, name_log)


# 为单个obj生成col
name_in = 'E:/research/dataset/grasp/ShapeNet/ShapeNetSem/test/1028b32dc1873c2afe26a3ac360dbd4/1028b32dc1873c2afe26a3ac360dbd4.obj'
name_out = name_in.replace('.obj', '_col.obj')
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log)
