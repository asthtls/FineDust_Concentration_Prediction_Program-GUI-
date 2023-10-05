# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:40:26 2022

@author: JEY
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
from netCDF4 import Dataset
import os 
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("chap2_gui2.ui")
form_main = uic.loadUiType(form)[0]

class MyWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Predict')
        self.exit_btn.clicked.connect(self.exit)
        
        self.x_test_variables = []
        self.y_test_variables = []
        self.choose_data = ''
        self.choose_method =''

        self.x_left_btn.clicked.connect(self.x_clicked_left_button)
        self.x_right_btn.clicked.connect(self.x_clicked_right_button)

        self.y_left_btn.clicked.connect(self.y_clicked_left_button)
        self.y_right_btn.clicked.connect(self.y_clicked_right_button)

        # data_method - radio btn click
        self.all_radio_btn.clicked.connect(self.check_data_method)
        self.zero_radio_btn.clicked.connect(self.check_data_method)
        self.cell_radio_btn.clicked.connect(self.check_data_method)

        # choose train method - 회귀, 딥러닝 , cnn 방식 선택 radio btn
        self.linear_radio_btn.clicked.connect(self.check_train_method)
        self.svr_radio_btn.clicked.connect(self.check_train_method)
        self.kernel_radio_btn.clicked.connect(self.check_train_method)
        self.gauss_radio_btn.clicked.connect(self.check_train_method)
        self.tree_radio_btn.clicked.connect(self.check_train_method)
        self.random_radio_btn.clicked.connect(self.check_train_method)
        self.neural_regression_radio_btn.clicked.connect(self.check_train_method)
        self.dnn_radio_btn.clicked.connect(self.check_train_method)
        self.rnn_radio_btn.clicked.connect(self.check_train_method)
        self.lstm_radio_btn.clicked.connect(self.check_train_method)
        self.gru_radio_btn.clicked.connect(self.check_train_method)
        self.conv1_radio_btn.clicked.connect(self.check_train_method)


    def get_parameter(self, x_test_path, y_test_path):
        nc_x = Dataset(x_test_path[0])
        tmp_x = nc_x.variables.keys()

        nc_y = Dataset(y_test_path[0])
        tmp_y = nc_y.variables.keys()
        cnt_x = 0
        cnt_y = 0
        for i in tmp_x:
            self.xtest_list.insertItem(cnt_x,i)
            cnt_x+=1
        for i in tmp_y:
            self.ytest_list.insertItem(cnt_y,i)
            cnt_y+=1
        


    def x_clicked_left_button(self):
        self.x_move_current_item(self.xtest_list, self.x_selected_view)
    
    def x_clicked_right_button(self):
        self.x_move_current_item(self.x_selected_view, self.xtest_list)
    
    def y_clicked_left_button(self):
        self.y_move_current_item(self.ytest_list, self.y_selected_view)

    def y_clicked_right_button(self):
        self.y_move_current_item(self.y_selected_view, self.ytest_list)
    
    def x_move_current_item(self, src, dst):
        item = src.currentItem()
        if item:
            row = src.currentRow()
            dst.addItem(src.takeItem(row))

            item = item.text()
            if item in self.x_test_variables:
                self.x_test_variables.remove(item)
            else:
                self.x_test_variables.append(item)
    
    def y_move_current_item(self, src, dst):
        item = src.currentItem()
        if item:
            row = src.currentRow()
            dst.addItem(src.takeItem(row))

            item = item.text()
            if item in self.y_test_variables:
                self.y_test_variables.remove(item)
            else:
                self.y_test_variables.append(item)

    def check_data_method(self):
        if self.all_radio_btn.isChecked():
            self.choose_data = 'all'
        elif self.zero_radio_btn.isChecked():
            self.choose_data = 'zero'
        elif self.cell_radio_btn.isChecked():
            self.choose_data = 'cell'

    def check_train_method(self):
        if self.linear_radio_btn.isChecked():
            self.choose_method = 'linear'
        elif self.svr_radio_btn.isChecked():
            self.choose_method = 'svr'
        elif self.kernel_radio_btn.isChecked():
            self.choose_method = 'kernel'
        elif self.gauss_radio_btn.isChecked():
            self.choose_method = 'gauss'
        elif self.tree_radio_btn.isChecked():
            self.choose_method = 'tree'
        elif self.random_radio_btn.isChecked():
            self.choose_method = 'randomforest'
        elif self.neural_regression_radio_btn.isChecked():
            self.choose_method = 'neural_regression'
        elif self.dnn_radio_btn.isChecked():
            self.choose_method = 'dnn'
        elif self.rnn_radio_btn.isChecked():
            self.choose_method = 'rnn'
        elif self.lstm_radio_btn.isChecked():
            self.choose_method = 'lstm'
        elif self.gru_radio_btn.isChecked():
            self.choose_method = 'gru'
        elif self.conv1_radio_btn.isChecked():
            self.choose_method = 'segnet'


    def get_data(self):
        return self.choose_data, self.choose_method, self.x_test_variables, self.y_test_variables

    def exit(self):
         self.close()

def gui_second(x_test_path, y_test_path):
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.get_parameter(x_test_path, y_test_path)
    myWindow.show()
    app.exec_()

    choose_data, choose_method, x_test_variables, y_test_variables = myWindow.get_data()
    choose_data = choose_data.upper()
    choose_method = choose_method.upper()
    x_test_variables = sorted(x_test_variables)

    return choose_data, choose_method, x_test_variables, y_test_variables


