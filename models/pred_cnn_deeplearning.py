
def pred_cnn_deeplearning(model_path, choose, x_test, y_test):
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error
    model_rmse = 0
    model_r2 = 0

    x_test = np.expand_dims(x_test, axis=0)
    
    choose = choose.upper()
    model = tf.keras.models.load_model(model_path)

    if choose =="CELL":
        y_pred = model.predict(x_test)
        y_pred = y_pred.ravel()
        model_r2 = r2_score(y_test, y_pred)
        model_rmse = mean_squared_error(y_test, y_pred) ** 0.5

        return model_rmse,model_r2, y_pred
    else:
        print("error data")


   
