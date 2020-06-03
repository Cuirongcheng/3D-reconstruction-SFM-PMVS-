import numpy as np
import os
from reconstruction import rec_config

while True:
    f = input('请输入三维点云的文件名：')+'.ply'
    # fd = os.listdir('../../models')
    # if f in fd:
    #     print('您输入的文件名已存在，请重新输入！')
    # else:
    #读取三维点坐标和d颜色信息
    points = np.float32(np.load(rec_config.image_dir + '/Structure.npy'))
    colors = np.load(rec_config.image_dir + '/Colors.npy')
    filename = f
    points =np.hstack([points.reshape(-1,3), colors.reshape(-1,3)])

    np.savetxt(filename, points, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    \n
    '''

    with open(filename,'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num = len(points)))
        f.write(old)
    break

print('点云存储完成！')