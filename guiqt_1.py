# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:21:52 2022

@author: JEY
"""
import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
import natsort

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("gui_1.ui")
form_main = uic.loadUiType(form)[0]

class MyWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Learning')
        self.xtrain_btn.clicked.connect(self.xtrain_open)
        self.ytrain_btn.clicked.connect(self.ytrain_open)
        self.xtest_btn.clicked.connect(self.xtest_open)
        self.ytest_btn.clicked.connect(self.ytest_open)
        self.exit_btn.clicked.connect(self.exit)
        self.x_train_path = []
        self.x_test_path = []
        self.y_train_path = []
        self.y_test_path = []
         
            
    def xtrain_open(self):
        global filename
        filename = QFileDialog.getOpenFileNames(self, 'Open File','')
        for file in filename[0]:
            dir, file = os.path.split(file)
        
            self.xLabel.append(str(file))
            file_path = dir + '/' + file
        
            self.x_train_path.append(file_path)

    def xtest_open(self):
        global filename
        filename = QFileDialog.getOpenFileNames(self, 'Open File','')
        
        for file in filename[0]:
            dir, file = os.path.split(file)
            
            self.xLabel_2.append(str(file))
            file_path = dir + '/' + file

            self.x_test_path.append(file_path)


    def ytrain_open(self):
        global filename
        filename = QFileDialog.getOpenFileNames(self, 'Open File','')
        
        for file in filename[0]:
            dir, file = os.path.split(file)
            
            self.yLabel.append(str(file))
            
            file_path = dir + '/' + file
            self.y_train_path.append(file_path)


    def ytest_open(self):
        global filename
        filename = QFileDialog.getOpenFileNames(self, 'Open File','')
        
        for file in filename[0]:
            dir, file = os.path.split(file)
            
            self.yLabel_2.append(str(file))

            file_path = dir + '/' + file
            self.y_test_path.append(file_path)
            
    def exit(self):
        self.close()
        
    def get_data(self):
        return self.x_train_path, self.x_test_path, self.y_train_path, self.y_test_path
       
# 첫 번째 챕터 : data load
def gui_frist():
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()

    x_train_path, x_test_path, y_train_path, y_test_path  = myWindow.get_data()

    x_train_path = natsort.natsorted(x_train_path)
    x_test_path = natsort.natsorted(x_test_path)
    y_train_path = natsort.natsorted(y_train_path)
    y_test_path = natsort.natsorted(y_test_path)

    return x_train_path, x_test_path, y_train_path, y_test_path
    