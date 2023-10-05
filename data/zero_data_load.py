# 0을 제거한 데이터 하나씩 return
import numpy as np

## train function
def zero_index_load(train_data):
    max_len = 0

    for i in range(len(train_data)):
        tmp_len = 0
        tmp_index = []
        tmp_index2 = []
        for j in range(len(train_data[0])):
            if train_data[i][j] == 0:
                tmp_len += 1
                tmp_index.append(j)
            
            elif train_data[i][j] != 0:
                tmp_index2.append(j)
        
        if tmp_len >= max_len:
            max_len = tmp_len
            zero_index = tmp_index
            exist_index = tmp_index2
        else:
            pass
        
    return zero_index,exist_index

def zero_train_load(train_data,length, zero_index):
    return_data = []
    for i in range(length):
        tmp_data = np.delete(train_data[i], zero_index)

        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.vstack((return_data, tmp_data))
    return return_data

def zero_test_load(test_data,length, zero_index):
    return_data = []
    for i in range(length):
        tmp_data = np.delete(test_data, zero_index)

        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.vstack((return_data, tmp_data))
    return np.array(return_data)


def pred_zero_index(train_data):
    max_len = 0
    for i in range(len(train_data[0])): # x 데이터의 특성만큼 
        tmp_len = 0
        tmp_index = []
        tmp_index2 = []
        for j in range(len(train_data)):  # 해당 데이터의 수 ex) 5494
            if train_data[j][i]== 0:
                tmp_len += 1
                tmp_index.append(j)
            
            elif train_data[j][i] != 0:
                tmp_index2.append(j)

        if tmp_len >= max_len:
            max_len = tmp_len
            zero_index = tmp_index
            exist_index = tmp_index2
        else:
            pass
    return zero_index, exist_index

def pred_zero_train_load(train_data,length, zero_index):
    return_data = []
    train_data = np.transpose(train_data)

    for i in range(length):
        tmp_data = np.delete(train_data[i], zero_index)

        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.vstack((return_data, tmp_data))

    return_data = np.transpose(return_data)
    return return_data

def pred_zero_test_load(test_data,length, zero_index):
    return_data = []
    for i in range(length):
        tmp_data = np.delete(test_data, zero_index)

        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.vstack((return_data, tmp_data))
    return np.array(return_data)






    
