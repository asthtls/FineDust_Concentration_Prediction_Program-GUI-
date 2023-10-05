import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tf_deeplearning import * 
from tf_cnn_deeplearning import * 
from tf_regression import * 
from all_data_load import *
from zero_data_load import * 
from cell_data_load import * 
from guiqt_1 import * 
from guiqt_2 import * 
from guiqt_3 import * 
import time

def create_folder():
    import os 
    model_path = './model/'
    log_path ='./log/'
    folder_name = ['LINEAR','SVR','KERNEL','GAUSS','TREE','RANDOMFOREST','NEURAL','DNN','RNN','LSTM','GRU','SEGNET']

    if os.path.isdir(model_path):
        pass
    else:
        os.mkdir(model_path)
    
    if os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)

    for i in range(len(folder_name)):
        if os.path.isdir(model_path + folder_name[i]):
            pass
        else:
            os.mkdir(model_path  + folder_name[i])
    
    for i in range(len(folder_name)):
        if os.path.isdir(log_path + folder_name[i]):
            pass
        else:
            os.mkdir(log_path + folder_name[i])


def data_load(choose_data, x_train_path, x_test_path, y_train_path, y_test_path,x_data_name, y_data_name):
    choose_data = choose_data.upper()
    exist_index = []
    zero_index  = []
    if choose_data == 'ALL':
        for i in range(len(x_data_name)):
            x_train_t= all_data_load(x_train_path, x_data_name[i])
            x_test_t = all_data_load(x_test_path, x_data_name[i])

            if i == 0:
                x_train = x_train_t
                x_test = x_test_t
                continue 
            else:
                pass

            x_train = np.vstack((x_train, x_train_t))
            x_test = np.vstack((x_test, x_test_t))

        for i in range(len(y_data_name)):
            y_train = all_data_load(y_train_path, y_data_name[i])
            y_test = all_data_load(y_test_path,y_data_name[i])

        x_train = np.transpose(x_train)
        x_test = np.transpose(x_test)
        
        return x_train, x_test, y_train, y_test,exist_index, zero_index 
    
    elif choose_data == 'ZERO':
        for i in range(len(x_data_name)):
            x_train_t= all_data_load(x_train_path, x_data_name[i])
            x_test_t = all_data_load(x_test_path, x_data_name[i])

            if i == 0:
                x_train = x_train_t
                x_test = x_test_t
            else:
                x_train = np.vstack((x_train, x_train_t))
                x_test = np.vstack((x_test, x_test_t))
                
        for i in range(len(y_data_name)):
            y_train = all_data_load(y_train_path, y_data_name[i])
            y_test = all_data_load(y_test_path,y_data_name[i])
        
        train_zero_index,train_exist_index = zero_index_load(x_train)
        test_zero_index, test_exist_index = zero_index_load(x_test)

        x_train = zero_train_load(x_train, len(x_data_name), train_zero_index)
        x_test = zero_train_load(x_test, len(x_data_name), test_zero_index)
        y_train = zero_test_load(y_train, len(y_data_name), train_zero_index)
        y_test = zero_test_load(y_test, len(y_data_name), test_zero_index)

        x_train = np.transpose(x_train)
        x_test = np.transpose(x_test)
        return x_train, x_test, y_train, y_test,test_exist_index,test_zero_index 

    elif choose_data == 'CELL':
        # x_data 전체 불러오기 0제거 x
        for i in range(len(x_data_name)):
            x_train_data = cell_data_load(x_train_path, x_data_name[i])
            x_test_data = cell_data_load(x_test_path, x_data_name[i])

            
            if i == 0:
                tmp_x_train = x_train_data
                tmp_x_test = x_test_data
                # x_train_data.shape, x_test_data.shape (83, 5494) (36, 5494)
            else:
                tmp_x_train = np.dstack((tmp_x_train, x_train_data))
                tmp_x_test = np.dstack((tmp_x_test, x_test_data))

        for i in range(len(tmp_x_train)):
            tmp_len, tmp_index, tmp_index2 = cell_zero_index(tmp_x_train[i])
        
            if i == 0:
                max_len = tmp_len
                zero_index = tmp_index
                exist_index = tmp_index2
                # tmp_len : 3958 : 0 제거 길이. tmp_index : 0인 index, tmp_index2 : 0이 아닌 값이 존재하는 index
            else:
                max_len = np.vstack((max_len, tmp_len))
                zero_index = np.vstack((zero_index, tmp_index))
                exist_index = np.vstack((exist_index, tmp_index2))
        
        print("CELL 2 index : ", max_len.shape, zero_index.shape, exist_index.shape)
        # x_data 0을 제거하고 불러오기
        for i in range(len(x_data_name)):
            x_train_t = cell_data_load(x_train_path, x_data_name[i])
            x_test_t = cell_data_load(x_test_path, x_data_name[i])
            tmp_x_train = cell_delete_load(x_train_t,len(x_train_path), zero_index[i],len(x_train_path))
            tmp_x_test = cell_delete_load(x_test_t, len(x_test_path), zero_index[i],len(x_test_path))

            if i == 0:
                x_train= tmp_x_train
                x_test = tmp_x_test
            else:
                x_train = np.dstack((x_train, tmp_x_train))
                x_test = np.dstack((x_test, tmp_x_test))

        # y_data load 
        for i in range(len(y_data_name)): 
            y_train_data = cell_data_load(y_train_path, y_data_name[i])
            y_test_data = cell_data_load(y_test_path, y_data_name[i])
            y_train = cell_delete_load(y_train_data, len(y_train_data), zero_index[i],len(y_train_path))
            y_test = cell_delete_load(y_test_data, len(y_test_path), zero_index[i], len(y_test_path))
 
        return x_train, x_test, y_train, y_test, exist_index,zero_index

def train_method(train_choose, choose_data, choose_method, x_train, x_test, y_train, y_test, data_len, model_store, exist_index,zero_index):
    train_choose = train_choose.upper()
    choose_data = choose_data.upper()
    choose_method = choose_method.upper()
    model_ypred = []
    model_ytest = []
    model_rmse = 0
    model_r2 = 0

    if choose_data == "ALL" or choose_data == "ZERO":
        if train_choose == "DEEPLEARNING":
            model_rmse, model_r2, model_ypred = tf_deeplearning(choose_data, choose_method, x_train, x_test, y_train, y_test,model_store, 1)
        elif train_choose == "REGRESSION":
            model_rmse, model_r2, model_ypred = tf_regression(choose_data, choose_method, x_train, x_test, y_train, y_test,model_store,1) 
    elif choose_data == "CELL":
        if train_choose == "DEEPLEARNING":
            for i in range(len(exist_index[0])): # 0데이터를 제거하고 남은 수만큼 반복
                tmp_ypred = tf_deeplearning(choose_data, choose_method, x_train[i], x_test[i], y_train[i], y_test, model_store, exist_index[0][i])

                if i == 0:
                    model_ypred = tmp_ypred
                    model_ytest = y_test[i]
                else:
                    model_ypred = np.vstack((model_ypred, tmp_ypred))
                    model_ytest = np.vstack((model_ytest, y_test[i]))

            model_ypred = np.reshape(model_ypred, (len(model_ypred) * len(model_ypred[0])))
            model_ytest = np.reshape(model_ytest, (len(model_ytest) * len(model_ytest[0])))
            model_rmse = mean_squared_error(model_ytest, model_ypred) ** 0.5
            model_r2 = r2_score(model_ytest, model_ypred)
        elif train_choose == "REGRESSION":

            for i in range(len(exist_index[0])):
                tmp_ypred = tf_regression(choose_data, choose_method, x_train[i], x_test[i], y_train[i], y_test[i], model_store, exist_index[0][i])
                
                if i == 0:
                    model_ypred = tmp_ypred
                    model_ytest = y_test[i]
                else:
                    model_ypred = np.vstack((model_ypred, tmp_ypred))
                    model_ytest = np.vstack((model_ytest, y_test[i]))
        
            model_ypred = np.reshape(model_ypred, (len(model_ypred) * len(model_ypred[0])))
            model_ytest = np.reshape(model_ytest, (len(model_ytest) * len(model_ytest[0])))
            model_rmse = mean_squared_error(model_ytest, model_ypred) ** 0.5
            model_r2 = r2_score(model_ytest, model_ypred)
        elif train_choose == "CNN_DEEPLEARNING":
            model_rmse, model_r2, model_ypred = tf_cnn_deeplearning(choose_method, x_train, x_test, y_train, y_test, model_store)

    return np.array(model_rmse), np.array(model_r2), np.array(model_ypred)


create_folder()

x_train_path, x_test_path, y_train_path, y_test_path = gui_frist()

x_data_name, y_data_name,choose_data, choose_method, model_store = gui_second(x_train_path, y_train_path)

exist_index = []
zero_index = []
train_choose = ''

x_train, x_test, y_train, y_test,exist_index, zero_index = data_load(choose_data,x_train_path,  x_test_path,y_train_path,y_test_path, x_data_name, y_data_name)

# train 방법론 선택
if choose_method == "LINEAR" or choose_method == "SVR" or choose_method =="KERNEL" or choose_method == "GAUSS" or choose_method == "TREE" or choose_method == "RANDOMFOREST" or choose_method == "NEURAL":
    train_choose = 'regression'
elif choose_method == "DNN" or choose_method == "RNN" or choose_method == "LSTM" or choose_method == "GRU":
    train_choose = 'deeplearning'
elif choose_method == "SEGNET":
    train_choose = 'cnn_deeplearning'

# 
if choose_data == "cell" and train_choose == "cnn_deeplearning":
    print("cnn_data")
    pass
elif choose_data == "cell":
    x_train = np.transpose(x_train, (1,0,2))
    x_test = np.transpose(x_test, (1,0,2))
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)
time_start = time.time()

rmse, r2,y_pred = train_method(train_choose,choose_data,choose_method,x_train, x_test, y_train, y_test, len(x_data_name),model_store,exist_index,zero_index)
time_end = time.time() - time_start

model_corr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0,1]  # 상관계수 제곱 계산

print(model_corr, model_corr**2)
gui3_start(rmse, r2,y_test,y_pred, time_end)

