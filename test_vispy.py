import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys


def show_in_vispy(path):
    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()


    # # generate data  使用随机数据的话把这块反注释掉
    # pos = np.random.normal(size=(100000, 3), scale=0.2)
    # print(pos)
    # # one could stop here for the data generation, the rest is just to make the
    # # data look more interesting. Copied over from magnify.py
    # centers = np.random.normal(size=(50, 3))
    # indexes = np.random.normal(size=100000, loc=centers.shape[0]/2.,
    #                            scale=centers.shape[0]/3.)
    # indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
    # scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
    # pos *= scales
    # pos += centers[indexes]
    # print(pos)
    # scatter = visuals.Markers()
    # scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    #
    #
    # #
    # #
    # # # 使用 kitti 数据， n*3
    # # img_id = 17  # 2，3 is not able for pcl;
    # # path = r'D:\KITTI\Object\training\velodyne\%06d.bin' % img_id  ## Path ## need to be changed
    # # points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    # #
    # #
    #

    # 读取点云
    # with open(path,"r") as f:
    #     lines = f.readlines()
    #     for i in range(len(lines)):
    #         lines[i] = list(map(float, lines[i].strip("\n").split(" ")))
    #     lines = np.array(lines)
    #     print(len(lines))

    with open(path,"r") as f:
        lines = f.readlines()
        lines = lines[13:]
        points = np.ones((len(lines),3))
        colors = []
        for i in range(len(lines)):
            points[i,:3] = list(map(float, lines[i].strip("\n").split(" ")[:3]))
            colors.append(tuple(list(map(float, lines[i].strip("\n").split(" ")[-3:]))))
        colors = np.array(colors)/255
        # colors[:,3] = 0.5
        # for i in range(len(colors)):
        #     colors[i] = tuple(colors[i])
        print(colors)
        print(points)
        # print(lines)

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(points[:,:3], edge_color=None, face_color=colors, size=4)

    view.add(scatter)
    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)


    if sys.flags.interactive != 1:
        vispy.app.run()

if __name__ == '__main__':
    show_in_vispy("D:\\04261.ply")