# all_data_input
# 전체 데이터를 0 제외하지 않고 하나의 파일만 테스트로 해서 return 

from netCDF4 import Dataset
import numpy as np	

def all_data_load(file_path,input_name): 
    return_data = [] # return train_data
    
    for i in range(len(file_path)):
        ncfile = Dataset(file_path[i])
        tmp_data = ncfile.variables[input_name][:]
        tmp_data = np.array(tmp_data).flatten()
        
        if i == 0:
            return_data = tmp_data
        else:
            return_data = np.concatenate((return_data, tmp_data), axis=0)

    return return_data
    
def pred_all_data_load(file_path, input_name):
    test_data = []
    return_data = []

    for i in range(len(file_path)):
        ncfile = Dataset(file_path[i])
        for j in range(len(input_name)):
            tmp_data = ncfile.variables[input_name[j]][:]
            tmp_data = np.array(tmp_data).flatten()

            if j == 0:
                test_data = tmp_data
            else:
                test_data = np.vstack((test_data, tmp_data))

        if i == 0:
            return_data = test_data
        else:
            return_data = np.dstack((return_data, test_data))

    return return_data
