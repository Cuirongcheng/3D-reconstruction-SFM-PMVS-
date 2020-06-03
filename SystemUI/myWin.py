import sys
from SystemUI.startupWin import *
from SystemUI.mainWin import *
from PyQt5.QtWidgets import *
import datetime
from PyQt5 import QtCore
import time
from calibration import calibration
from reconstruction import rec_config
from reconstruction.spare import sfmui
from reconstruction.dense import PMVSui,Dense_filterui
import numpy as np
import vispy.scene
from vispy.scene import visuals


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str) #定义一个发送str的信号
    def write(self, text):
      self.textWritten.emit(str(text))

class Startwin(QMainWindow, Ui_MainWindow):
    # welcome_page = []
    def __init__(self,parent = None):
        super(Startwin, self).__init__(parent)
        self.setupUi(self)
        # self.welcome_page.append(self.label)
        # self.welcome_page.append(self.pushButton)

        self.addWidgets()
        # 设置窗体无边框
        # self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
        # 设置文本框及按钮

    def addWidgets(self):
        self.setnametip = QtWidgets.QLabel(self.centralwidget)
        self.setnametip.setText("请输入工程文件名")
        self.setnametip.setGeometry(QtCore.QRect(400, 350, 180, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.setnametip.setFont(font)
        self.setnametip.setObjectName("setnametip")
        self.setnametip.close()

        self.LineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.LineEdit.setGeometry(QtCore.QRect(350, 390, 250, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.LineEdit.setFont(font)
        self.LineEdit.setObjectName("LineEdit")
        self.LineEdit.close()

        self.namesubButton = QtWidgets.QPushButton(self.centralwidget)
        self.namesubButton.setGeometry(QtCore.QRect(400, 438, 70, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.namesubButton.setFont(font)
        self.namesubButton.setObjectName("namesubButton")
        self.namesubButton.setText("确认")
        self.namesubButton.close()

        self.cancleButton = QtWidgets.QPushButton(self.centralwidget)
        self.cancleButton.setGeometry(QtCore.QRect(480, 438, 70, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.cancleButton.setFont(font)
        self.cancleButton.setObjectName("cancleButton")
        self.cancleButton.setText("取消")
        self.cancleButton.close()

        # # 加到数组中
        # self.welcome_page.append(self.setnametip)
        # self.welcome_page.append(self.LineEdit)
        # self.welcome_page.append(self.namesubButton)
        # self.welcome_page.append(self.cancleButton)

        # 绑定事件
        self.pushButton.clicked.connect(self.setname)
        self.cancleButton.clicked.connect(self.canclesetname)
        self.namesubButton.clicked.connect(self.subname)

    def getprojectname(self):
        return self.projectname



    def setname(self):
        self.pushButton.close()
        self.cancleButton.show()
        self.namesubButton.show()
        self.setnametip.show()
        self.LineEdit.show()

    def canclesetname(self):
        self.pushButton.show()
        self.cancleButton.close()
        self.namesubButton.close()
        self.setnametip.close()
        self.LineEdit.close()

    def subname(self):
        Input_text = self.LineEdit.text()
        if Input_text !="":
            self.projectname = Input_text
            # print(Input_text)
        else:
            self.projectname = "Project-" + str(datetime.datetime.now()).split()[0]  # 默认工程文件名
        print("projectname:",self.projectname)
        # 保存文件名到txt文件中
        with open("../project_name.txt", "w") as f:
            f.write(self.projectname)
            f.close()
        Start.close()
        mainwin.show()


class MainWin(QDialog,Ui_Dialog):
    def __init__(self,parent = None):
        super(MainWin, self).__init__(parent)
        self.setupUi(self)

        # 下面将输出重定向到textBrowser中
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.pushButton.clicked.connect(self.open_cali_images)
        self.textBrowser.close()

        self.addWidgets()


    def addWidgets(self):
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_4.setText("相机标定完毕，请选择重建图片")
        self.label_4.close()

        self.reconsbtn = QtWidgets.QPushButton(self)
        self.reconsbtn.setGeometry(QtCore.QRect(340, 320, 240, 41))
        self.reconsbtn.setObjectName("reconsbtn")
        self.reconsbtn.setText("点击选择重建图片文件夹")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.reconsbtn.setFont(font)
        self.reconsbtn.clicked.connect(self.recons)
        self.reconsbtn.close()

        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_4")
        self.label_5.setText("稀疏重建完成，现在开始稠密重建")
        self.label_5.close()

        self.previewsparsebtn = QtWidgets.QPushButton(self)
        self.previewsparsebtn.setGeometry(QtCore.QRect(340, 380, 240, 41))
        self.previewsparsebtn.setObjectName("sparsebtn")
        self.previewsparsebtn.setText("点击显示稀疏重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.previewsparsebtn.setFont(font)
        self.previewsparsebtn.clicked.connect(self.showSparse)
        self.previewsparsebtn.close()

        self.densereconsbtn = QtWidgets.QPushButton(self)
        self.densereconsbtn.setGeometry(QtCore.QRect(340, 320, 240, 41))
        self.densereconsbtn.setObjectName("reconsbtn")
        self.densereconsbtn.setText("点击开始稠密重建")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.densereconsbtn.setFont(font)
        self.densereconsbtn.clicked.connect(self.denserecons)
        self.densereconsbtn.close()

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_4")
        self.label_6.setText("重建完成，还需过滤掉一些杂点")
        self.label_6.close()

        self.filterbtn = QtWidgets.QPushButton(self)
        self.filterbtn.setGeometry(QtCore.QRect(355, 320, 240, 41))
        self.filterbtn.setObjectName("filterbtn")
        self.filterbtn.setText("点击开始点云过滤")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.filterbtn.setFont(font)
        self.filterbtn.clicked.connect(self.pointfilter)
        self.filterbtn.close()


        self.previewdensebackbtn = QtWidgets.QPushButton(self)
        self.previewdensebackbtn.setGeometry(QtCore.QRect(355, 380, 240, 41))
        self.previewdensebackbtn.setObjectName("previewdensebackbtn")
        self.previewdensebackbtn.setText("点击显示稠密重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.previewdensebackbtn.setFont(font)
        self.previewdensebackbtn.clicked.connect(self.showDensewithback)
        self.previewdensebackbtn.close()


        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(440, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_4")
        self.label_7.setText("重建完成！")
        self.label_7.close()

        self.finishbtn = QtWidgets.QPushButton(self)
        self.finishbtn.setGeometry(QtCore.QRect(370, 320, 240, 41))
        self.finishbtn.setObjectName("reconsbtn")
        self.finishbtn.setText("完成")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.finishbtn.setFont(font)
        self.finishbtn.clicked.connect(self.recfinish)
        self.finishbtn.close()

        self.showfinalbtn = QtWidgets.QPushButton(self)
        self.showfinalbtn.setGeometry(QtCore.QRect(370, 380, 240, 41))
        self.showfinalbtn.setObjectName("showfinalbtn")
        self.showfinalbtn.setText("点击显示足部重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.showfinalbtn.setFont(font)
        self.showfinalbtn.clicked.connect(self.showfinal)
        self.showfinalbtn.close()



    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def show_in_vispy(self, path):
        # Make a canvas and add simple view
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()


        # 读取点云
        with open(path, "r") as f:
            lines = f.readlines()
            lines = lines[13:]
            points = np.ones((len(lines), 3))
            colors = []
            for i in range(len(lines)):
                points[i, :3] = list(map(float, lines[i].strip("\n").split(" ")[:3]))
                colors.append(tuple(list(map(float, lines[i].strip("\n").split(" ")[-3:]))))
            colors = np.array(colors) / 255

        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(points[:, :3], edge_color=None, face_color=colors, size=4)

        view.add(scatter)
        view.camera = 'turntable'  # or try 'arcball'

        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=view.scene)

        if sys.flags.interactive != 1:
            vispy.app.run()

    def open_cali_images(self):
        self.cali_img_dir = QFileDialog.getExistingDirectory(self,"选取文件夹","../")  # 起始路径

        calibration.cali_img_dir = self.cali_img_dir
        self.calithead = cali()
        self.calithead.signal.connect(self.callback1)

        self.calithead.start()
        self.label_3.setText("正在进行相机标定，请等候片刻")
        self.label_3.setGeometry(QtCore.QRect(290, 230, 381, 51))
        self.textBrowser.show()
        self.pushButton.close()


    def recons(self):
        self.rec_img_dir = QFileDialog.getExistingDirectory(self, "选取文件夹", "../")  # 起始路径

        rec_config.image_dir = self.rec_img_dir
        self.recthead = rec()
        self.recthead.signal.connect(self.callback2)
        self.recthead.start()
        self.textBrowser.show()
        # self.label_4.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_4.setText("正在进行稀疏重建，请等候片刻")
        self.reconsbtn.close()

    def denserecons(self):
        PMVSui.CMVS.image_dir = self.rec_img_dir
        # print(self.rec_img_dir)
        self.denserecthead = denserec()
        self.denserecthead.signal.connect(self.callback3)
        self.denserecthead.start()
        # self.textBrowser.show()
        # self.label_5.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_5.setText("正在进行稠密重建，请等候片刻")
        self.densereconsbtn.close()
        self.previewsparsebtn.close()

    def pointfilter(self):
        self.pfilt = pointfilt()
        self.modelpath = QFileDialog.getExistingDirectory(self, "选取路径", "../")  # 起始路径
        self.pfilt.signal.connect(self.callback4)
        self.pfilt.start()
        # self.textBrowser.show()
        # self.label_6.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_6.setText("正在进行过滤多余点云，请等候片刻")
        self.filterbtn.close()
        self.previewdensebackbtn.close()

    def recfinish(self):
        mainwin.close()
        Start.show()
        Start.pushButton.show()
        Start.setnametip.close()
        Start.cancleButton.close()
        Start.namesubButton.close()
        Start.LineEdit.close()

    def showSparse(self):
        sfmui.save_sparse()
        self.show_in_vispy(self.rec_img_dir + '/sparse.ply')

    def showDensewithback(self):
        # # 将ply转换成txt
        # with open("../reconstruction/dense/pmvs/models/option-0000.ply", "r+") as f:
        #     lines = f.readlines()
        # with open('../reconstruction/dense/pmvs/models/dense.txt', "w") as f:
        #     for i in range(13, len(lines)):
        #         if lines[i] != '':
        #             f.write(" ".join(lines[i].split()[:3]) + "\n")
        self.show_in_vispy('../reconstruction/dense/pmvs/models/option-0000.ply')

    def showfinal(self):
        self.show_in_vispy(self.modelpath+ '/' + Start.getprojectname() + '.ply')

    def callback1(self,i):
        # self.label_3.setText("aa")
        print("=============相机标定完成=============")
        # self.mythead.terminate()
        mainwin.close()
        mainwin.show()

        self.label_3.close()
        self.pushButton.close()
        # self.textBrowser.close()
        self.label_4.show()
        self.reconsbtn.show()
        self.label_2.setText("*------稀疏重建--------*")

    def callback2(self,i):
        # self.label_3.setText("aa")
        print("=============稀疏重建完成=============")
        mainwin.close()
        mainwin.show()

        self.label_4.close()
        # self.reconsbtn.close()
        # self.textBrowser.close()
        self.label_5.show()
        self.densereconsbtn.show()
        self.previewsparsebtn.show()
        self.label_2.setText("*------稠密重建--------*")

    def callback3(self,i):
        # self.label_3.setText("aa")
        print("=============稠密重建完成=============")
        mainwin.close()
        mainwin.show()

        self.label_5.close()
        # self.densereconsbtn.close()
        # self.previewsparsebtn.close()
        # self.textBrowser.close()
        self.label_2.setText("*------过滤点云--------*")
        self.label_6.show()
        self.filterbtn.show()
        self.previewdensebackbtn.show()

    def callback4(self,i):
        # self.label_3.setText("aa")
        print("=============点云过滤完成=============")
        mainwin.close()
        mainwin.show()

        self.label_6.close()
        # self.filterbtn.close()
        # self.previewdensebackbtn.close()
        # self.textBrowser.close()
        self.label_2.setText("*------完成--------*")
        self.label_7.show()
        self.finishbtn.show()
        self.showfinalbtn.show()

    def getmodelpath(self):
        return self.modelpath



class cali(QtCore.QThread): # 建立相机标定子线程
    signal = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(cali,self).__init__()

    def run(self):
        print("===========现在开始相机标定===========")
        calibration.Zhang()
        # time.sleep(2)
        self.signal.emit(True)


class rec(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(rec, self).__init__()

    def run(self):
        print("===========现在开始稀疏重建===========")
        sfmui.sfm_rec()
        # time.sleep(2)
        self.signal.emit(True)

class denserec(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(denserec,self).__init__()

    def run(self):
        print("===========现在开始稠密重建===========")
        PMVSui.denserec()
        time.sleep(3)
        self.signal.emit(True)

class pointfilt(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(pointfilt,self).__init__()

    def run(self):
        print("===========现在开始点云过滤===========")
        # Dense_filterui.pointfilter(mainwin.getmodelpath(),Start.getprojectname())
        # time.sleep(2)
        self.signal.emit(True)

if __name__ =='__main__':
    app = QApplication(sys.argv)
    Start = Startwin()
    Start.show()
    mainwin = MainWin()

    sys.exit(app.exec_())