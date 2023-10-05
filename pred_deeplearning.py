
def pred_deeplearning(model_path, choose, method, x_test, y_test):
    import tensorflow as tf
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import numpy as np

    print(np.array(x_test).shape, np.array(y_test).shape)
    model_rmse = 0
    model_r2 = 0

    choose = choose.upper()
    method = method.upper()

    if method == "RNN" or method == "LSTM" or method == "GRU":
        x_test = np.reshape(x_test, (len(x_test), 1, len(x_test[0])))    
    else:
        pass
    model = tf.keras.models.load_model(model_path)

    if choose == "ALL" or "ZERO":    
        y_pred = model.predict(x_test)
        model_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        model_r2 = r2_score(y_test, y_pred)

        y_pred = np.squeeze(y_pred)
        return model_rmse, model_r2, y_pred
    elif choose =="CELL":
        y_pred = model.predict(x_test)
        return y_pred


   