import os

def is_skin(point_rgb):
    R = 0
    G = 1
    B = 2
    if((point_rgb[R] > 95 and point_rgb[G]>40 and point_rgb[B] > 20 and
                point_rgb[R] - point_rgb[B]>50 and point_rgb[R] - point_rgb[G]>15)):
        return True
    return False


def pointfilter(modelpath,projectname):
    l = 0 # 点数统计
    with open("../reconstruction/dense/pmvs/models/option-0000.ply","r+") as f:
       lines = f.readlines()
       # print(len(lines))
       for i in range(13,len(lines)):
           point_rgb = lines[i].strip('\n').split(" ")[6:]
           point_rgb[:] = map(int, point_rgb)
           # 判断该点是否为肤色
           # print(is_skin(point_rgb))
           if is_skin(point_rgb):
               l += 1
               continue
           lines[i] = ''
    # print(modelpath)
    # 保存txt用于其他建模软件
    with open(modelpath + '/' + projectname + '.txt', "w") as f:
        # lines[2] = 'element vertex ' + str(l) + '\n'
        # for line in lines:
        #     f.write(str(line))
        for i in range(13, len(lines)):
            if lines[i] !='':
                f.write(" ".join(lines[i].split()[:3]) + "\n")
    # 保存ply用于显示
    with open(modelpath + '/' + projectname + '.ply', "w") as f:
        lines[2] = 'element vertex ' + str(l) + '\n'
        for line in lines:
            f.write(str(line))
        # for i in range(13, len(lines)):
        #     f.write(" ".join(lines[i].split()[:3]) + "\n")

    print("点云保存成功！")


