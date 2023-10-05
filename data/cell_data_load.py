import numpy as np
def cell_data_load(file_path, input_name):
    from netCDF4 import Dataset

    return_data =[]

    for i in range(len(file_path)):
        ncfile = Dataset(file_path[i])
        tmp_data = ncfile.variables[input_name][:]
        tmp_data = np.array(tmp_data).flatten()

        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.vstack((return_data, tmp_data))

        # print(return_data.shape)
    return return_data

def cell_zero_index(train_data):
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
    return max_len, zero_index, exist_index

def cell_delete_load(data, length,zero_index, data_len):
    return_data = []
    if data_len == 1:
        for i in range(length):
            tmp_data = np.delete(data, zero_index)

            if i == 0:
                return_data = tmp_data
            else:
                return_data = np.vstack((return_data, tmp_data))
    else:
        for i in range(length):
            tmp_data = np.delete(data[i], zero_index)

            if i == 0:
                return_data = tmp_data
            else:
                return_data = np.vstack((return_data, tmp_data))
    return return_data

