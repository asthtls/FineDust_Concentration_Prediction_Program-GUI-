import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np



def tf_deeplearning(choose, method, X_train, X_test, Y_train,Y_test, model_store, exist_index):
    method = method.upper()
    choose = choose.upper()
    if choose == 'CELL':
        model_store ='./model/'+method +'/'+model_store+str(exist_index)
        batch_size = 8

    elif choose =="ZERO":
        model_store = './model/'+method+'/'+model_store
        batch_size = 64
    elif choose== "ALL":
        model_store = './model/'+method+'/'+model_store
        batch_size= 128
    else:
        print('error')

    model_rmse = 0
    model_r2 = 0
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5)

    if method =="DNN":
        # 256, 128, 64, 1
        input_size = (len(X_train[0]),)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=(input_size),activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
    elif method == "RNN":
        # 
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(32, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
        ])


    # INPUT_DATA = batchsize, sequence_layer, feature x == 32, ?, 5
    elif method == "LSTM":
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
        

    elif method == "GRU":
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.GRU(32, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
        ])


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='MSE',
                metrics=['accuracy'])

    if choose == "CELL":
        
        model.fit(X_train, Y_train, epochs=75, batch_size=batch_size, callbacks=callback)

        model.save(model_store + '.h5')

        y_pred = model.predict(X_test)
        
        return y_pred

    elif choose ==  "ALL" or choose == "ZERO":    
        
        model.fit(X_train, Y_train, epochs=100, batch_size = batch_size, callbacks=callback)
        y_pred = model(X_test)
        model.save(model_store + '.h5')
        y_pred = np.squeeze(y_pred)
        Y_test = np.squeeze(Y_test)
        model_rmse = mean_squared_error(Y_test, y_pred) ** 0.5
        model_r2 = r2_score(Y_test, y_pred)

        return model_rmse, model_r2, y_pred

    else:
        print("No search choose")
