
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic,QtWidgets
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
import numpy as np
import os 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("gui_3.ui")
form_main = uic.loadUiType(form)[0]
data_max = 0

class MyWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Learning')
        self.exit_btn.clicked.connect(self.exit)

    def get_data(self,rmse, r2, y_test,y_pred,time):
        self.rmse_view.setPlainText(str(rmse))
        self.r2_view.setPlainText(str(r2))
        self.train_time.setPlainText(str(time))
        #for draw graph
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.scatter_plot_layout.addWidget(self.canvas)
        self.canvas = FigureCanvas(self.fig)
        ax = self.fig.add_subplot(111)
        y_pred = y_pred.ravel()
        y_test = y_test.ravel()

        if max(y_pred) > max(y_test):
            data_max = max(y_pred)
        else:
            data_max = max(y_test)
        data_max = math.ceil(data_max)
        ax.set_title('Scatter plot')
        ax.set_ylabel('y_test')
        ax.set_xlabel('y_pred')
        ax.set_xlim(0, data_max)
        ax.set_ylim(0, data_max)
        ax.scatter(x=y_pred, y=y_test)
        b, a = np.polyfit(y_pred, y_test, deg=1)

        xseq = np.linspace(0, data_max, num=100)
        ax.plot(xseq,a + b * xseq, color='k',lw=2.5)


    def exit(self):
         self.close()
         
def gui3_start( rmse, r2,y_test, y_pred, time):
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.get_data(rmse, r2, y_test, y_pred, time)
    myWindow.show()
    app.exec_()

