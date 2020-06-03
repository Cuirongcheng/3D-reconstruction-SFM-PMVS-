import cv2
import numpy as np
import os
import math
from mayavi import mlab
from tqdm import tqdm
from scipy.optimize import least_squares
from reconstruction import rec_config
import glob
import shutil
from scipy.sparse import lil_matrix
from calibration import calibration

cali_img_dir = calibration.cali_img_dir

def extract_feathers(image_names):
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()

    keypoints_for_all = []
    descriptors_for_all = []
    colors_for_all = []

    for image_name in tqdm(image_names):
        image = cv2.imread(image_name)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # 添加该图片的关键点信息
        keypoints_for_all.append(keypoints)
        descriptors_for_all.append(descriptors)

        # 获取关键点颜色信息
        colors = np.zeros((len(keypoints), 3))
        for i, keypoint in enumerate(keypoints):
            p = keypoint.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)
    return np.array(keypoints_for_all), np.array(descriptors_for_all), np.array(colors_for_all)


# 特征点匹配
def match_feathers(kp1,kp2,des1, des2):
    # 用FLANNE算法对两张照片进行关键点匹配，返回这两张照片的匹配对
    # FLANN参数设计
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 核密度树数量
    # search_params = dict(checks=50)  # 遍历次数

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        # RANSAC随机采样一致
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,100.0)
        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        i = 0
        j = 0
        while (j < len(mask)):
            if mask[j] == 0:
                good.remove(good[i])
                j += 1
            else:
                i += 1
                j += 1

    else:
        print("not have enough matches!")


    return np.array(good)


# 对相邻图片特征匹配
def match_all_feather(keypoints_for_all,descriptors_for_all):
    # 只需相邻图片进行匹配 1-2 2-3 ...N-1 - N 一共N-1对匹配
    matches_for_all = []
    for i in tqdm(range(len(descriptors_for_all) - 1)):
        matches = match_feathers(keypoints_for_all[i],keypoints_for_all[i+1],descriptors_for_all[i], descriptors_for_all[i + 1])
        matches_for_all.append(matches)

    return np.array(matches_for_all)

# ============Part 2==========
# 计算本质矩阵和外参
def find_transform(K, p1, p2):

    # 根据匹配点求本征矩阵，使用RANSAC进一步消除失配点
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)  # 基于五点算法 mask用来标记RANSAC计算得来的内点 只有内点才被用于后续步骤
    # 分解本征矩阵
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, K, mask)

    return R, T, mask


# 三维重建
def reconstruction(K, R1, T1, R2, T2, p1, p2):
    # 设置两个相机的投影矩阵[R T]，且转为float型数据 triangulatePoints函数只支持float
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))

    proj1[0:3, 0:3] = np.float32(R1)  # 将R1赋值给proj1的对应位置（前三行三列）
    proj1[:, 3] = np.float32(T1.T)  # 将T1的转置赋值给proj1的对应位置（第四列）
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3] = np.float32(T2.T)

    fk = np.float32(K)

    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)

    # p1 p2 原本Nx2 转置为2*N
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

    structure = []
    for i in range(len(s[0])):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到“正常”的点坐标
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])
    return np.array(structure), proj2


# 获取匹配点坐标
def get_matched_points(p1, p2, matches):
    out_p1 = np.asarray([p1[m.queryIdx].pt for m in matches])  # queryIndex 查询图像中描述符的索引
    out_p2 = np.asarray([p2[m.trainIdx].pt for m in matches])

    return out_p1, out_p2


# 获取匹配点颜色信息
def get_matched_colors(c1, c2, matches):
    color1 = np.asarray([c1[m.queryIdx] for m in matches])
    color2 = np.asarray([c2[m.trainIdx] for m in matches])

    return color1, color2


# 获取内点的坐标
def maskout_point(p, mask):
    p_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p_copy.append(p[i])

    return np.array(p_copy)


# 获取内点颜色信息
def maskout_color(c, mask):
    c_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            c_copy.append(c[i])

    return np.array(c_copy)


# ============Part 3==========
# 初始化点云
def init_structure(K, keypoints_for_all, colors_for_all, matches_for_all):
    # 找到前两幅图的关键点坐标和相应的颜色信息
    p1, p2 = get_matched_points(keypoints_for_all[0], keypoints_for_all[1],
                                matches_for_all[0])  # matches_for_all[0] 为第一幅图和第二幅图之间的匹配对
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

    # 计算外参
    if find_transform(K, p1, p2):
        R, T, mask = find_transform(K, p1, p2)
    else:
        R, T, mask = np.array([]), np.array([]), np.array([])

    # 对前两幅图进行三维重建
    p1 = maskout_point(p1, mask)
    p2 = maskout_point(p2, mask)
    colors = maskout_color(c1, mask)

    R0 = np.eye(3, 3)  # 设置第一个相机的变换矩阵，作为剩下摄像机矩阵变换的基准
    T0 = np.zeros((3, 1))
    proj0 = np.zeros((3, 4))  # 第一个相机的投影矩阵
    proj0[0:3, 0:3] = np.identity(3)  # 将R1赋值给proj1的对应位置（前三行三列）
    proj0[:, 3] = np.ones(3)
    fk = np.float32(K)
    proj0 = np.dot(fk, proj0)
    projections = [proj0]

    structure, proj = reconstruction(K, R0, T0, R, T, p1, p2)
    projections.append(proj) # 加入第二幅图片的投影矩阵

    rotations = [R0, R]
    motions = [T0, T]


    correspond_struct_idx = []  # correspond_struct_idx[i][j]表示第i幅图的第j个特征点对应的空间点在点云中的索引值（空间点云中第几个点？）
    for kp in keypoints_for_all:
        correspond_struct_idx.append(np.ones(len(kp)) * -1)  # 初始化correspond_struct_idx的大小 图片数*相应图片特征点数目 各元素初始值为-1

    # 填写前两幅图特征点的索引
    idx = 0  # 初始化空间点云索引值
    for i, match in enumerate(matches_for_all[0]):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1

    return structure, correspond_struct_idx, colors, rotations, motions, projections


# 获取空间点和像素点用于solvePnPRansac
def get_objpoints_and_imgpoints(matches, struct_indices, structure, keypoints):
    # matches是第i幅图与第i+1幅图之间的匹配，struct_indices是第i幅图特征点对应的空间点索引，structure为已有点云，keypoints为第i+1幅图的关键点
    obj_points = []
    img_points = []

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]

        if struct_idx < 0:
            continue
        obj_points.append(structure[int(struct_idx)])
        img_points.append(keypoints[int(train_idx)].pt)

    return np.array(obj_points), np.array(img_points)


# 点云融合
def fusion_structure(matches, struct_indict1, struct_indict2, structure, new_structure, colors, new_colors):
    # matches：图i与图i+1之间的匹配
    # struct_indict1：图i的特征点在点云中的索引，struct_indict2：图i+1的特征点在点云中的索引
    # structure:原有的点云 new_structure:新生成的三维点
    # colors: 原有点云各点的颜色，new_colors：新生成点云各点的颜色
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indict1[query_idx]

        # 如果新生成的三维点已存在，则告诉图i+1是哪个空间点
        if struct_idx >= 0:
            struct_indict2[train_idx] = struct_idx
            continue

        # 如果新生成的三维点不存在于原有点云中，则加入新点
        structure = np.append(structure, [new_structure[i]], axis=0)
        colors = np.append(colors, [new_colors[i]], axis=0)
        struct_indict1[query_idx] = struct_indict2[train_idx] = len(structure) - 1

    return structure, colors, struct_indict1, struct_indict2



# ============Part 4==========
# BA优化部分

# 根据重映射误差优化三维点
# 1 删除误差大的点
def get_3dpoints_v1(objp, imgp, R, T, K):
    p, J = cv2.projectPoints(objp.reshape(1, 1, 3), R, T, K, np.array([]))
    p = p.reshape(2)
    e = imgp - p  # 计算误差
    if e[0] > 1.0 or e[1] > 1.0:  # 误差太大则不要这个点
        return None
    return objp  # 只返回误差满足要求的点

# 2 对整个点云进行BA优化
def reproject_error(rotations, motions, K, struct_index, key_points_for_all, structure):
    # 这里需要矩阵
    errors = 0
    num = 0
    #这里需要的是旋转向量
    for i in tqdm(range(len(struct_index))):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        point3d_idxs = struct_index[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]

        for j in range(len(point3d_idxs)):
            point3d_idx = int(point3d_idxs[j])
            if point3d_idx < 0:
                continue
            num += 1
            objp = structure[point3d_idx].reshape(1, 1, 3)
            imgp = key_points[j].pt
            p, J = cv2.projectPoints(objp, r, t, K, np.array([]))
            p = p.reshape(2)
            e = imgp - p  # 计算误差
            error = (e[0] + e[1]) / 2
            errors += error
    avg_error = errors / num
    print("点云数目：{},平均误差：{}".format(len(structure), avg_error, ))


def delete_error_point(rotations, motions, K, struct_index, key_points_for_all, structure):
    # BA优化
    for i in tqdm(range(len(struct_index))):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        point3d_idxs = struct_index[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]

        # 对该图片包含的空间点进行优化
        for j in range(len(point3d_idxs)):
            point3d_idx = int(point3d_idxs[j])
            if point3d_idx < 0:
                continue
            new_point = get_3dpoints_v1(structure[point3d_idx], key_points[j].pt, r, t, K)
            structure[point3d_idx] = new_point

    return structure

def fun(param, n_cameras, n_points, struct_index, key_points_for_all, K):
    # param: rotations, motions, 以及 points的扁平列表【rotation*3, motion*3,point*3】
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    pro_point = []
    rotations = param[:n_cameras * 3].reshape((n_cameras, 3))
    motions = param[n_cameras * 3:n_cameras * 6].reshape((n_cameras, 3))
    structure = param[n_cameras * 6:].reshape((n_points, 3))

    for i in range(len(struct_index)):
        # 获取每张图片特征点、对应的点云索引、旋转向量、平移矩阵
        point3d_idxs = struct_index[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]

        for j in range(len(point3d_idxs)):
            point3d_idx = int(point3d_idxs[j])
            if point3d_idx < 0:
                continue
            objp = structure[point3d_idx].reshape(1, 1, 3)
            imgp = key_points[j].pt
            p, J = cv2.projectPoints(objp, r, t, K, np.array([]))
            p = p.reshape(2)
            e = imgp - p
            pro_point.append(e[0])
            pro_point.append(e[1])
    return np.array(pro_point)

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    # TODO: 雅可比稀疏矩阵
    m = camera_indices.size * 2  # n_observation * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def read_data(indexs):
    camera_indices = np.array([])
    point_indices= np.array([])

    for i in  range(len(indexs)):
        for j in range(len(indexs[i])):
            if indexs[i][j] > -1:
                camera_indices = np.append(camera_indices, i)
                point_indices = np.append(point_indices, indexs[i][j])


    return camera_indices, point_indices

def bundle_adjustment(structure, indexs, motions, rotat, keypoints_for_all, K):
    camera_indices, point_indices = read_data(indexs)
    n_cameras = len(motions)
    n_points = len(structure)

    # 这里需要的是向量
    rotations = []
    for i in range(len(rotat)):
        R, _ = cv2.Rodrigues(rotat[i])  # 旋转矩阵转向量
        rotations.append(R)
    rotations = np.array(rotations)

    x0 = np.hstack((rotations.ravel(), motions.ravel(), structure.ravel()))

    f0 = fun(x0, n_cameras, n_points, indexs, keypoints_for_all, K)

    # TODO: 这里可以输出原误差

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, indexs, keypoints_for_all, K))
    structure = np.array(res.x[n_cameras*6:]).reshape((-1,3))

    # TODO: 这里可以输出现有误差
    return structure


def save_bundle_rd_out(structure, K, rotations, motions, colors, correspond_struct_idxs, keypoints_for_all):
    # print("处理数据")
    lines = []
    # header = "# Bundle file v0.3"
    lines.append("# Bundle file v0.3")
    num_camera = len(rotations)
    num_point = len(structure)
    lines.append(str(num_camera)+" "+str(num_point))

    # print('1')
    for i in range(num_camera):
        focal_length = 0.5 * (K[0][0] + K[1][1])
        R = rotations[i]
        T = motions[i]
        lines.append(str(focal_length) + " 0 0")
        lines.append(str(R[0][0]) + " " + str(R[0][1]) + " " + str(R[0][2]))
        lines.append(str(R[1][0]) + " " + str(R[1][1]) + " " + str(R[1][2]))
        lines.append(str(R[2][0]) + " " + str(R[2][1]) + " " + str(R[2][2]))
        lines.append(str(T[0][0]) + " " + str(T[1][0]) + " " + str(T[2][0]))
    # print("2")
    for i in range(num_point):
        lines.append(str(structure[i][0]) + " " + str(structure[i][1]) + " " + str(structure[i][2]))
        lines.append(str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]))

        # get detailed information of points
        # 1. the number of cameras in which can observe the point
        # 2. the corresponding information about projection points of the point in images  camera index, keypoint index, x/y coordinante
        count = 0  # counter the number of camera
        info = ""
        for a in range(len(correspond_struct_idxs)):
            if i in correspond_struct_idxs[a]:
                for b in range(len(correspond_struct_idxs[a])):
                    if correspond_struct_idxs[a][b] == i:
                        count += 1
                        info = info + str(a) + " "
                        info = info + str(b) + " "
                        info = info + str(keypoints_for_all[a][b].pt[0]) + " " + str(keypoints_for_all[a][b].pt[1]) + " "
                        break

        lines.append(str(count) + " " + info)

    with open("../reconstruction/dense/pmvs/bundle.rd.out","w") as f:
    # with open("../dense/pmvs/bundle.rd.out", "w") as f:
        # TODO: GUI更改文件夹位置3
        print("开始写入")
        s = "\n"
        f.write(s.join(lines))

def save_sparse():
    image_dir = rec_config.image_dir

    points = np.float32(np.load(image_dir + '/Structure.npy'))
    colors = np.load(image_dir + '/Colors.npy')
    filename = 'sparse.ply'
    points = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])

    np.savetxt(image_dir + '/' + filename, points, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header

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

    with open(image_dir + '/' + filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(points)))
        f.write(old)

def sfm_rec():
    image_dir = rec_config.image_dir + '/'
    image_names = glob.glob(image_dir+'*.jpg')  # 读取图片本身的名字
    image_names = sorted(image_names)
    with open("../project_name.txt", "r") as f:
        project_name = f.read()
    # TODO: GUI更改文件夹位置1
    # with open("../../project_name.txt", "r") as f:
    #     project_name = f.read()

    K = np.load('../calibration/camera_params/' + project_name + '/K.npy')
    # TODO: GUI更改文件夹位置2
    # K = np.load('../../calibration/camera_params/' + project_name + '/K.npy')

    # 提取特征点、特征匹配
    print('提取所有图片特征点')
    keypoints_for_all, descriptors_for_all, colors_for_all = extract_feathers(image_names)
    # print(colors_for_all)
    print('匹配所有图片特征')
    matches_for_all = match_all_feather(keypoints_for_all, descriptors_for_all)
    for i in range(len(matches_for_all)):
        print(len(matches_for_all[i]),end=' ')

    # 初始化点云
    print('\n初始化点云')
    structure, correspond_struct_idx, colors, rotations, motions, projections = init_structure(K, keypoints_for_all, colors_for_all,
                                                                                  matches_for_all)
    print("初始化点云数目:",len(structure))
    # 增量方式添加剩余点云
    print('增量方式添加剩余点云')
    for i in tqdm(range(1, len(matches_for_all))):
        # 获取第i幅图中匹配点的空间三维坐标，以及第i+1幅图匹配点的像素坐标

        obj_points, img_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i], structure,
                                                             keypoints_for_all[i + 1])
        # solvePnPRansac得到第i+1个相机的旋转和平移
        # 在python的opencv中solvePnPRansac函数的第一个参数长度需要大于7，否则会报错
        # 这里对小于7的点集做一个重复填充操作，

        if len(obj_points) < 7:
            while len(img_points) < 7:
                obj_points = np.append(obj_points, [obj_points[0]], axis=0)
                img_points = np.append(img_points, [img_points[0]], axis=0)

        # 得到第i+1幅图相机的旋转向量和位移矩阵
        _, r, T, _ = cv2.solvePnPRansac(obj_points, img_points, K, np.array([]))
        R, _ = cv2.Rodrigues(r)  # 将旋转向量转换为旋转矩阵
        rotations.append(R)
        motions.append(T)

        # 根据R T进行重建
        p1, p2 = get_matched_points(keypoints_for_all[i], keypoints_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])

        new_structure, new_proj = reconstruction(K, rotations[i], motions[i], R, T, p1, p2)

        projections.append(new_proj)
        # 点云融合
        structure, colors, correspond_struct_idx[i], correspond_struct_idx[i + 1] = fusion_structure(matches_for_all[i],
                                                                                                     correspond_struct_idx[
                                                                                                         i],
                                                                                                     correspond_struct_idx[
                                                                                                         i + 1],
                                                                                                     structure,
                                                                                                     new_structure,
                                                                                                     colors, c1)
        print("新生成点云数" ,len(new_structure) , "第" , i , "次融合后点云数" , len(structure))

    print(len(structure))
    # BA优化
    print('删除误差较大的点')
    structure = delete_error_point(rotations, motions, K, correspond_struct_idx, keypoints_for_all, structure)


    # 由于经过bundle_adjustment的structure，会产生一些空的点（实际代表的意思是已被删除）
    # 修改各图片中关键点的索引
        # 修改点云中的None为 -1
    for i in range(len(structure)):
        if math.isnan(structure[i][0]):
            structure[i] = -1

        # 修改各图片中的索引
    for a in range(len(correspond_struct_idx)):
        for b in range(len(correspond_struct_idx[a])):
            if correspond_struct_idx[a][b] != -1:
                if structure[int(correspond_struct_idx[a][b])][0] == -1:
                    correspond_struct_idx[a][b] = -1
                else:
                    correspond_struct_idx[a][b] -= (np.sum(structure[:int(correspond_struct_idx[a][b])] == -1)/3)

    i = 0
    # 删除那些为空的点
    while i < len(structure):
        if structure[i][0] == -1:
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1
    reproject_error(rotations, motions, K, correspond_struct_idx, keypoints_for_all, structure)
    print('BA优化')
    motions = np.array(motions)
    rotations = np.array(rotations)
    structure = bundle_adjustment(structure, correspond_struct_idx, motions, rotations, keypoints_for_all, K)

    reproject_error(rotations, motions, K, correspond_struct_idx, keypoints_for_all, structure)

    #保存Bundle.rd.out
    print("点云已生成，正在保存.out文件")
    # 旋转向量转化为旋转矩阵
    Rotations = np.empty((len(rotations), 3, 3))
    for i in range(len(rotations)):
        R, _ = cv2.Rodrigues(rotations[i])
        Rotations[i] = R

    save_bundle_rd_out(structure, K, Rotations, motions, colors, correspond_struct_idx, keypoints_for_all)

    np.save(image_dir + 'Structure', structure)
    np.save(image_dir + 'Colors', colors)
    np.save(image_dir + 'Projections', projections)
    # np.save(image_dir + 'Rotations', rotations)
    # np.save(image_dir + 'Motions', motions)
    # np.save(image_dir + 'Indexs', correspond_struct_idx)
    #
    # for i in range(len(keypoints_for_all)):
    #     keypoints_for_all[i] = cv2.KeyPoint_convert(keypoints_for_all[i])
    # np.save(image_dir + 'Keypoints_for_all', keypoints_for_all)