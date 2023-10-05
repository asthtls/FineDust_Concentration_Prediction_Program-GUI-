import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
import sys 
import numpy as np


def cnn_model(input_shape):        
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 3,strides=1,padding='same', input_shape=(input_shape),activation='relu'))
    model.add(tf.keras.layers.Conv1D(64, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv1D(128, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(tf.keras.layers.Conv1D(256, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(tf.keras.layers.Conv1D(512, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(tf.keras.layers.Conv1D(1024, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.UpSampling1D(2))
    model.add(tf.keras.layers.Conv1D(512, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.UpSampling1D(2))
    model.add(tf.keras.layers.Conv1D(256, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.UpSampling1D(2))
    model.add(tf.keras.layers.Conv1D(64, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv1D(32, 3, 1,padding='same',activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    
    return model


def tf_cnn_deeplearning(method, x_train, x_test, y_train, y_test, model_store):

    input_shape = (len(x_train[0]), len(x_train[0][0]))

    model_store = './model/'+method+'/'+model_store

    rmse = 0
    r2 = 0
    # learning_rate = 0.001

    if method == "SEGNET":
        model = cnn_model(input_shape) 

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss='MSE',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=100, batch_size=4)

        print(model.summary())
        model.save(model_store+'.h5')
        # rmse, r2 
        y_pred = model.predict(x_test)

        y_pred = y_pred.flatten()
        y_test = y_test.flatten()

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        return rmse, r2, y_pred 
    else:
        sys.exit("try another way cnn")
