# ============张正友标定法获取内参==============
import cv2
import glob
import numpy as np
from tqdm import tqdm
# from calibration import cali_config
import os
import shutil

cali_img_dir_root = './cali_images'
cali_img_filename = 'i8p'

cali_img_dir = cali_img_dir_root + '/'+ cali_img_filename


def Zhang():
    image_size = (0, 0)
    # 设置棋盘板长宽
    chessboard_size = (9, 6)

    # 定义数组存储检测到的点
    obj_points = []  # 真实世界中的三维坐标
    img_points = []  # 图片平面的二维坐标

    ####准备目标坐标 （0，0，0），（1，0，0）...（9，6，0）
    # 设置世界坐标下的坐标值
    # 假设棋盘正好在x-y平面上，z直接取零，从而简化初始步骤
    # objp包含的是10*7每一角点的坐标
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)  # 9*6个三维坐标
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # 标定图片所在目录
    # cali_img_dir = cali_config.cali_img_dir

    # 读取图片，使用glob文件名管理工具
    # calibration_paths = glob.glob("../cali_images/i8p/*.JPG")

    # 读取每张图片路径
    calibration_paths = os.listdir(cali_img_dir)
    for i in range(len(calibration_paths)):
        calibration_paths[i] = cali_img_dir + '/' + calibration_paths[i]

    # 对每张图片，识别出角点，记录世界物体坐标和图像坐标
    for img_path in tqdm(calibration_paths):
        # tqdm是进度条，以了解距离处理上一个图像多长时间，还剩多少图像没有处理
        # 加载图片
        img = cv2.imread(img_path)
        # 照片太大 缩小一半 (缩小内参会变！！像素变小了 并不是对原图像处理)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        #
        # cv2.imshow('i',img)
        # cv2.waitKey(0)
        # 寻找角点将其存入corners(该图片9*6个角点坐标)，ret是找到角点的标志(True/False)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            # 检测到角点执行以下操作（一般都能检测到角点，除非图片不是规定的棋盘格）
            # 定义角点精准化迭代过程的终止条件 （包括精度和迭代次数）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0010)
            # 执行亚像素级角点检测
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners2)

        # 可视化角点
        # img = cv2.drawChessboardCorners(gray,(9,6),corners2,ret)
        # cv2.imshow('s',img)
        # cv2.waitKey(100)

    # 相机标定
    # 每张图片都有自己的旋转和平移矩阵 但是相机内参是畸变系数只有一组（因为相机没变，焦距和主心坐标是一样的）
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    # 保存参数
    K = K * 5
    K[2][2] = 1

    # 创建目录用于存放本次标定参数（文件名是唯一的）
    with open("../project_name.txt","r") as f:
        project_name = f.read()
    params_dir = '../calibration/camera_params/'+ project_name
    if os.path.exists(params_dir):
        shutil.rmtree(params_dir)
    os.mkdir(params_dir+"/")

    np.save(params_dir + '/K', K)
    print(K)
    # return True

    '''
    np.save(params_dir+'/ret', ret)
    np.save(params_dir+'/dist', dist)
    np.save("../camera_params/rvecs", rvecs)
    np.save("../camera_params/tvecs", tvecs)


    K=K*0.2
    K[2][2] = 1
    tot_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        tot_error += error

    print("total error: ", tot_error*5 / len(obj_points))

    '''
