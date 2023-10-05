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
from PyQt5.QtWidgets import QListWidgetItem
from netCDF4 import Dataset
import os 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("gui_2.ui")
form_main = uic.loadUiType(form)[0]

class MyWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Learning')
        self.exit_btn.clicked.connect(self.exit)
        self.x_train_variables = []
        self.y_train_variables = []


        self.choose_data = '' # 2-2. 데이터 방식 선택 cell, cnn, all, zero
        self.choose_method = '' # 2-3. 학습 방식 선택, dnn, randomforest 등
        self.model_name_store = '' # 2-4. 모델 저장 경로
        
        
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

        #finde model store 
        #self.model_store_btn.clicked.connect(self.model_store_open)
        self.model_btn.clicked.connect(self.button_event)
    def button_event(self):
        text = self.model_name.text() # line_edit text 값 가져오기
        self.model_name_store = text

    def get_parameter(self, x_train_path, y_train_path):
        nc_x = Dataset(x_train_path[0])
        tmp_x = nc_x.variables.keys()

        nc_y = Dataset(y_train_path[0])
        tmp_y = nc_y.variables.keys()
        cnt_x = 0
        cnt_y = 0
        for i in tmp_x:
            self.xtrain_list.insertItem(cnt_x,i)
            cnt_x+=1
        for i in tmp_y:
            self.ytrain_list.insertItem(cnt_y,i)
            cnt_y+=1
        
    def x_clicked_left_button(self):
        self.x_move_current_item(self.xtrain_list, self.x_selected_view)
    
    def x_clicked_right_button(self):
        self.x_move_current_item(self.x_selected_view, self.xtrain_list)
    
    def y_clicked_left_button(self):
        self.y_move_current_item(self.ytrain_list, self.y_selected_view)

    def y_clicked_right_button(self):
        self.y_move_current_item(self.y_selected_view, self.ytrain_list)

    def x_move_current_item(self, src, dst):
        item = src.currentItem()
        if item:
            row = src.currentRow()
            dst.addItem(src.takeItem(row))

            item = item.text()
            if item in self.x_train_variables:
                self.x_train_variables.remove(item)
            else:
                self.x_train_variables.append(item)
    
    def y_move_current_item(self, src, dst):
        item = src.currentItem()
        if item:
            row = src.currentRow()
            dst.addItem(src.takeItem(row))

            item = item.text()
            if item in self.y_train_variables:
                self.y_train_variables.remove(item)
            else:
                self.y_train_variables.append(item)
    


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
            self.choose_method = 'neural'
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
        return self.x_train_variables, self.y_train_variables, self.choose_data,self.choose_method, self.model_name_store

    def exit(self):
         self.close()

def gui_second(x_train_path, y_train_path):
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.get_parameter(x_train_path, y_train_path)
    myWindow.show()
    app.exec_()

    x_train_variables, y_train_variables, choose_data, choose_method, model_name_store = myWindow.get_data()
    x_train_variables = sorted(x_train_variables)
    choose_method = choose_method.upper()

    return x_train_variables, y_train_variables, choose_data, choose_method, model_name_store
    


