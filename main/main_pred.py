# model load 

# 2챕터 진행

# 1. 모델 load 
# 2. 데이터 input x, y
# 3. 데이터 특성 load x, y 데이터 특성
# 4. rmse, r2
# 5. 

from chap2_gui1 import * 
from chap2_gui2 import * 
from chap2_gui3 import * 
from all_data_load import *
from zero_data_load import * 
from cell_data_load import * 
import numpy as np
from pred_regression import * 
from pred_deeplearning import *
from pred_cnn_deeplearning import *  
from sklearn.metrics import r2_score, mean_squared_error
import time
# 모델 load 

def data_load(choose_data, x_test_path, y_test_path, x_data_name, y_data_name):
    choose_data = choose_data.upper()
    zero_index = []
    exist_index = []    
    if choose_data == 'ALL':
        x_test = pred_all_data_load(x_test_path, x_data_name)
        y_test = pred_all_data_load(y_test_path, y_data_name)

    
    elif choose_data == 'ZERO' or choose_data == "CELL":
        x_test_t = pred_all_data_load(x_test_path, x_data_name)
        y_test_t = pred_all_data_load(y_test_path, y_data_name)
        x_test_t = np.transpose(x_test_t)
        y_test_t = np.transpose(y_test_t)

        if len(x_test_path) == 1:
            x_test_t = np.reshape(x_test_t, (len(x_test_path), len(x_test_t), len(x_test_t[0])))
            y_test_t = np.reshape(y_test_t, (len(y_test_path), len(y_test_t)))

        # 데이터를 전달받고 전체 하나씩 셀처럼 전달해서 다시 리턴받는다. index데이터 반환 받기
        zero_index, exist_index = pred_zero_index(x_test_t[0])

        for i in range(len(x_test_t)):
            tmp_x = pred_zero_train_load(x_test_t[i], len(x_data_name), zero_index)
            tmp_y = pred_zero_test_load(y_test_t[i], len(y_data_name), zero_index)
            if i == 0:
                x_test = tmp_x
                y_test = tmp_y
            else:
                x_test = np.dstack((x_test, tmp_x))
                y_test = np.dstack((y_test, tmp_y))

    if choose_data == "ALL":
        x_test = np.transpose(x_test)
        y_test = np.transpose(y_test)
        if len(x_test_path) == 1:
            x_test = np.expand_dims(x_test, axis = 0)
            y_test = np.reshape(y_test, (len(y_test_path), len(y_test),1))
        else: pass
 
    if choose_data == "ZERO" or choose_data =="CELL":
        if len(x_test_path) == 1:
            x_test = np.reshape(x_test, (len(x_test_path), len(x_test), len(x_test[0])))
            y_test = np.reshape(y_test, (len(y_test_path), len(y_test)))
        else:
            x_test = np.transpose(x_test, (2,0,1))
            y_test = np.squeeze(y_test)
            y_test = np.transpose(y_test)

    return x_test, y_test,zero_index,exist_index

def model_load(train_choose, choose_data, choose_method, model_path, x_test,y_test):
    train_choose = train_choose.upper()
    choose_data = choose_data.upper()
    choose_method = choose_method.upper()
    model_rmse = []
    model_r2 = []
    model_ypred = []


    if train_choose == "DEEPLEARNING":
        if choose_data == "ALL" or choose_data == "ZERO":
            for i in range(len(x_test)):
                tmp_rmse, tmp_r2, tmp_ypred = pred_deeplearning(model_path[0], choose_data,choose_method, x_test[i], y_test[i])
                model_rmse.append(tmp_rmse)
                model_r2.append(tmp_r2)
                if i == 0:
                    model_ypred = tmp_ypred
                else:
                    model_ypred = np.dstack((model_ypred, tmp_ypred))
        elif choose_data =="CELL":
            for j in range(len(model_path)):
                tmp_ypred = pred_deeplearning(model_path[j], choose_data, choose_method,x_test[i], y_test[i])
                for j in range(len(model_path)):
                    cell_ypred.append(tmp_ypred)

                if i == 0:
                    model_ypred = cell_ypred
                else:
                    model_ypred = np.dstack((model_ypred, cell_ypred))
            if len(x_test) != 0:
                model_ypred = np.squeeze(model_ypred)
            model_ypred = np.transpose(model_ypred)
            for i in range(len(x_test)):
                tmp_r2 = r2_score(y_test[i],model_ypred[i])
                tmp_rmse = mean_squared_error(y_test[i], model_ypred[i]) 

                model_r2.append(tmp_r2)
                model_rmse.append(tmp_rmse)

    elif train_choose == "REGRESSION":
        if choose_data == "ALL" or choose_data == "ZERO":
            for i in range(len(x_test)):
                tmp_rmse, tmp_r2, tmp_ypred = pred_regression(model_path[0], choose_data, choose_method,x_test[i], y_test[i])

                model_rmse.append(tmp_rmse)
                model_r2.append(tmp_r2)
                if i == 0:
                    model_ypred = tmp_ypred
                else:
                    model_ypred = np.dstack((model_ypred, tmp_ypred))

        elif choose_data == "CELL":
            for i in range(len(x_test)):
                cell_ypred = []
                for j in range(len(model_path)):
                    tmp_ypred = pred_regression(model_path[j], choose_data,choose_method, x_test[i][j], y_test)
                    
                    cell_ypred.append(tmp_ypred)
                
                if i == 0:
                    model_ypred = cell_ypred 
                else:
                    model_ypred = np.dstack((model_ypred, cell_ypred))

            if len(x_test) != 1:
                model_ypred = np.squeeze(model_ypred)    
            model_ypred = np.transpose(model_ypred)
            for i in range(len(x_test)):
                tmp_r2 = r2_score(y_test[i],model_ypred[i])
                tmp_rmse = mean_squared_error(y_test[i], model_ypred[i]) 

                model_r2.append(tmp_r2)
                model_rmse.append(tmp_rmse)
    elif train_choose == "CNN_DEEPLEARNING":
        if choose_data == "CELL":
            for i in range(len(x_test)): # 데이터의 개수만큼 반복
                tmp_rmse, tmp_r2, tmp_ypred = pred_cnn_deeplearning(model_path[0], choose_data, x_test[i], y_test[i])

                model_rmse.append(tmp_rmse)
                model_r2.append(tmp_r2)
                if i == 0:
                    model_ypred = tmp_ypred
                else:
                    model_ypred = np.dstack((model_ypred, tmp_ypred))

    if choose_data == "ALL" or choose_data == "ZERO":
        if len(x_test) == 1:
            model_ypred = np.reshape(model_ypred,  (len(x_test), len(model_ypred)))
        else:
            model_ypred = np.squeeze(model_ypred)
            model_ypred = np.transpose(model_ypred)
    elif train_choose == "CNN_DEEPLEARNING" and choose_data == "CELL":
        if len(x_test) == 1:
            model_ypred = np.reshape(model_ypred,  (len(x_test), len(model_ypred)))
        else:
            model_ypred = np.squeeze(model_ypred)
            model_ypred = np.transpose(model_ypred)
    return model_rmse, model_r2, model_ypred


model_path, x_test_path, y_test_path = gui_frist()
choose_data, choose_method,x_data_name, y_data_name = gui_second(x_test_path, y_test_path)

if choose_method == "LINEAR" or choose_method == "SVR" or choose_method =="KERNEL" or choose_method == "GAUSS" or choose_method == "TREE" or choose_method == "RANDOMFOREST" or choose_method == "NEURAL":
    train_choose = 'regression'
elif choose_method == "DNN" or choose_method == "RNN" or choose_method == "LSTM" or choose_method == "GRU":
    train_choose = 'deeplearning'
elif choose_method == "SEGNET":
    train_choose = 'cnn_deeplearning'

x_test,y_test,zero_index, exist_index = data_load(choose_data, x_test_path, y_test_path,x_data_name, y_data_name)

start_time = time.time()
rmse, r2, y_pred = model_load(train_choose, choose_data, choose_method, model_path, x_test,y_test)
end_time = time.time() - start_time

for i in range(len(x_test_path)):
    print("현재 모델 : ", x_test_path[i])
    gui3_start(choose_data, x_test_path, x_data_name, exist_index, zero_index, rmse[i], r2[i], y_pred[i],end_time)

