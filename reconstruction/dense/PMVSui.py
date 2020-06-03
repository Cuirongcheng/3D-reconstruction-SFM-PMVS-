import os
import subprocess
import shutil
from reconstruction import rec_config
import numpy as np
import glob
import cv2
import reconstruction.spare.sfmui as sfm
# os.system是简单粗暴的执行cmd指令，如果想获取在cmd输出的内容，是没办法获到的
# 用os.popen的方法了，popen返回的是一个file对象，跟open打开文件一样操作了，r是以读的方式打开
# 三个指令一起执行会出错 调用os模块失败
# 使用subprocess模块，等待一个命令执行完成之后 再执行下一个命令


class CMVS:
    image_dir = rec_config.image_dir


    def cmvs(self):
        # 获取根路径
        root = "../reconstruction/dense"

        # 设置命令
        commend1 = root + "\cmvs.exe" + " " + root + "\pmvs\\"
        commend2 = root + "\genOption.exe" + " " + root + "\pmvs\\"
        commend3 = root + "\pmvs2.exe" + " " + root + "\pmvs\ option-0000"

        # 执行CMVS
        process = subprocess.Popen(commend1)
        process.wait()
        process = subprocess.Popen(commend2)
        process.wait()
        process = subprocess.Popen(commend3)
        process.wait()

    def prepare_visualize(self):
        # print(self.image_dir)
        dst_dir = '../reconstruction/dense/pmvs/visualize/'

        shutil.rmtree(dst_dir)  # empty dir.6+



        os.mkdir(dst_dir)

        #image_dir = self.image_dir + "New_images/"
        image_names = glob.glob(self.image_dir+'/*.jpg')
        image_names = sorted(image_names)

        for i,img in enumerate(image_names):
            shutil.copyfile(img,dst_dir + str(i).zfill(8) + '.jpg')
            # img = cv2.imread(img)
            # cv2.imwrite(dst_dir + str(i).zfill(8) + '.jpg', img)

    def prepare_txt(self):
        dst_dir = '../reconstruction/dense/pmvs/txt/'

        shutil.rmtree(dst_dir)  # empty dir
        os.mkdir(dst_dir)

        projections = np.load(self.image_dir + '/Projections.npy')

        #print(projections[0][0][0])
        #创建每张图片的txt文件
        for i in range(len(projections)):
            with open("../reconstruction/dense/pmvs/txt/"+str(i).zfill(8)+".txt","w") as f:
                f.write("CONTOUR\r"
                        +str(projections[i][0][0])+" "+str(projections[i][0][1])+" "+str(projections[i][0][2])+" "+str(projections[i][0][3])+"\r"
                        +str(projections[i][1][0])+" "+str(projections[i][1][1])+" "+str(projections[i][1][2])+" "+str(projections[i][1][3])+"\r"
                        +str(projections[i][2][0])+" "+str(projections[i][2][1])+" "+str(projections[i][2][2])+" "+str(projections[i][2][3])+"\r")

    # def move_model(self):
    #     while True:
    #         f = input('请输入三维点云的文件名：') + '.ply'
    #         fd = os.listdir('../../models')
    #         if f in fd:
    #             print('您输入的文件名已存在，请重新输入！')
    #         else:
    #             print(f)
    #             shutil.copyfile('./pmvs/models/option-0000.ply', '../../models/'+f)
    #             break
    #     print("点云保存成功！")

def denserec():

    print("开始稠密重建")
    # create CMVS objector
    My_CMVS = CMVS()
    My_CMVS.prepare_visualize()
    My_CMVS.prepare_txt()
    My_CMVS.cmvs()
    # My_CMVS.move_model()


