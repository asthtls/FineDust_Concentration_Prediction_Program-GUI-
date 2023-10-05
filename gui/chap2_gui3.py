
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic,QtWidgets
from netCDF4 import Dataset
import numpy as np
import os 
import pandas as pd

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("chap2_gui3.ui")
form_main = uic.loadUiType(form)[0]
data_max = 0

class MyWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Predict')
        self.exit_btn.clicked.connect(self.exit)

        self.csv_path = []
        self.csv_name_t = ''
        self.csv_ypred = []
        self.csv_shape= []

        # csv 저장 경로 버튼 클릭 경로 받기
        self.csv_path_btn.clicked.connect(self.csv_path_open)
        self.csv_name_btn.clicked.connect(self.csv_name_open)
        self.csv_store.clicked.connect(self.make_csv)

        # csv 저장 이름
    def csv_name_open(self):
        text = self.csv_name.text() # line_edit text 값 가져오기
        self.csv_name_t = text

    def csv_path_open(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.csv_path_label.addItem(str(folder))
        self.csv_path = str(folder)

    def make_csv(self):
        df = pd.DataFrame(self.csv_ypred)
        df.to_csv(self.csv_path + '/'+self.csv_name_t+'.csv', index=False)
        df2 = pd.DataFrame(np.transpose(self.csv_shape), columns=['x','y','y_pred'])
        df2.to_csv(self.csv_path + '/' +self.csv_name_t+"_shape.csv", index=False)
        
    def get_data(self,choose_data,  x_train_path,x_data_name, exist_index, zero_index, rmse, r2, y_pred,end_time):
        self.rmse_view.setPlainText(str(rmse))
        self.r2_view.setPlainText(str(r2))
        self.pred_time.setPlainText(str(end_time))
        
        ypred_nc = Dataset(x_train_path[0])
        ypred_nc = ypred_nc.variables[x_data_name[0]][:]
        ypred_nc = np.array(ypred_nc)
        ypred_shape = ypred_nc[0][0].shape
        ypred_shape = (ypred_shape[1], ypred_shape[0])
        choose_data = choose_data.upper()

        if choose_data == "ALL":
            # 전체 데이터 reshape 하면 끝남        
            y_pred = y_pred.reshape(ypred_shape)
            self.ypred_table.setRowCount(len(y_pred))
            self.ypred_table.setColumnCount(len(y_pred[0]))

            tmp_df = pd.DataFrame(columns=['x_index','y_index','ypred'])
            cnt = 1
            for i in range(len(y_pred)):
                for j in range(len(y_pred[0])):
                    self.ypred_table.setItem(i,j, QTableWidgetItem(str(y_pred[i][j])))
                    tmp = [[str(i+1)], [str(j+1)], [y_pred[i][j]]]
                    if cnt == 1:
                        tmp_df = tmp
                    else:
                        tmp_df = np.concatenate([tmp_df, tmp],axis=1)
                    cnt +=1

            self.csv_ypred = y_pred
            self.csv_shape = tmp_df

        elif choose_data == "ZERO":
            tmp_df = []
            cnt = 1
            ypred_len = len(exist_index) + len(zero_index)

            ypred_zero = np.zeros(ypred_len)

            for i in range(len(exist_index)):
                ypred_zero = np.insert(ypred_zero, exist_index[i], y_pred[i])

            ypred_zero = ypred_zero[:ypred_len]
            ypred_zero = ypred_zero.reshape(ypred_shape)

            self.ypred_table.setRowCount(len(ypred_zero))
            self.ypred_table.setColumnCount(len(ypred_zero[0]))
            for i in range(len(ypred_zero)):
                for j in range(len(ypred_zero[0])):
                    self.ypred_table.setItem(i,j, QTableWidgetItem(str(ypred_zero[i][j])))
                    tmp = [[str(i+1)], [str(j+1)], [ypred_zero[i][j]]]
                    if cnt == 1:
                        tmp_df = tmp
                    else:
                        tmp_df = np.concatenate([tmp_df, tmp],axis=1)
                    cnt +=1

            self.csv_ypred = ypred_zero
            self.csv_shape = tmp_df

        elif choose_data == "CELL":
            ypred_len = len(exist_index) + len(zero_index)
            ypred_zero = np.zeros(ypred_len)
            for i in range(len(exist_index)):
                ypred_zero = np.insert(ypred_zero, exist_index[i], y_pred[i])

            ypred_zero = ypred_zero[:ypred_len]
            ypred_zero = ypred_zero.reshape(ypred_shape)
            
            self.ypred_table.setRowCount(len(ypred_zero))
            self.ypred_table.setColumnCount(len(ypred_zero[0]))


            tmp_df = pd.DataFrame(columns=['x_index','y_index','ypred'])
            cnt = 1
            for i in range(len(ypred_zero)):
                for j in range(len(ypred_zero[0])):
                    self.ypred_table.setItem(i,j, QTableWidgetItem(str(ypred_zero[i][j])))
                    tmp = [[str(i+1)], [str(j+1)], [ypred_zero[i][j]]]
                    if cnt == 1:
                        tmp_df = tmp
                    else:
                        tmp_df = np.concatenate([tmp_df, tmp],axis=1)
                    cnt +=1

            self.csv_ypred = ypred_zero
            self.csv_shape = tmp_df
    def exit(self):
         self.close()
         
def gui3_start(choose_data, x_train_path,x_data_name ,exist_index, zero_index, rmse, r2,y_pred,end_time):
    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.get_data(choose_data, x_train_path,x_data_name, exist_index, zero_index,rmse, r2, y_pred,end_time)
    myWindow.show()
    app.exec_()

