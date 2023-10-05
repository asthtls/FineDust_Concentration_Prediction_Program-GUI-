def tf_regression(choose, method, X_train, X_test, Y_train, Y_test, model_store, exist_index):
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import joblib
    import sys
    import numpy as np
    
    if choose == 'CELL':
        model_store ='./model/'+method +'/'+model_store+str(exist_index)

    elif choose == "ALL" or choose =="ZERO":
        model_store = './model/'+method+'/'+model_store

    else:
        print('error')
        sys.exit('try another way_regression')
    mdl_rmse = 0
    mdl_r2 = 0

    if method == "LINEAR":
        from sklearn.linear_model import LinearRegression
        import statsmodels.api as sm

        X_t = sm.add_constant(X_train)
        sm_ols = sm.OLS(Y_train, X_t)
        ols = sm_ols.fit()
        model = LinearRegression()
        model.fit(X_train, Y_train.ravel())

    elif method == "SVR":
        from sklearn.svm import SVR
        model = SVR()
        model.fit(X_train, Y_train.ravel())

    elif method == "KERNEL":
        from kernel_regression import KernelRegression
        # from statsmodels import nonparametric
        # model = nonparametric()
        #KernelRidge(kernel ='linear', alpha=0.0)
        model = KernelRegression()
        model.fit(X_train, Y_train.ravel())

        if choose == "CELL":
            y_pred = model.predict(X_test)
            joblib.dump(model, model_store +'.pkl')
            return y_pred
        else:
            y_pred = model.predict(X_test)
            mdl_rmse = mean_squared_error(Y_test, y_pred) ** 0.5
            mdl_r2 = r2_score(Y_test, y_pred)
            joblib.dump(model, model_store +'.pkl')
            return mdl_rmse, mdl_r2, y_pred

    elif method == "GAUSS":
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor()
        model.fit(X_train, Y_train.ravel())

    elif method == "TREE":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(X_train, Y_train.ravel())

    elif method == "RANDOMFOREST":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X_train, Y_train.ravel())
        
    elif method == "NEURAL": # 추후 하이퍼파라미터 keras tuner 수정
        import tensorflow as tf
        
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='MSE',
        metrics=['accuracy'])

        if choose == "CELL":
            model.fit(X_train, Y_train, epochs=50, batch_size=8, callbacks=callback)
            y_pred = model.predict(X_test)
            model.save(model_store + '.h5')
            return y_pred
        else:
            model.fit(X_train, Y_train, epochs=100, batch_size = 64, callbacks=callback)
            y_pred = model(X_test)
            mdl_rmse = mean_squared_error(Y_test, y_pred) ** 0.5
            mdl_r2 = r2_score(Y_test, y_pred)

            model.save(model_store+'.h5')
            return mdl_rmse, mdl_r2, y_pred
    if choose == "CELL":
        if method =="LINEAR":
            y_pred = model.predict(X_test)

            # with open('Linear_summary_'+str(exist_index)+'.txt','w') as fh:
            #     fh.write(ols.summary().as_text())

            # with open('Linear_summary_'+str(exist_index)+'.csv', 'w') as fh:
            #     fh.write(ols.summary().as_csv())
            joblib.dump(model, model_store +'.pkl')
        else:
            y_pred = model.predict(X_test)
            joblib.dump(model, model_store +'.pkl')
        return y_pred
    elif choose ==  "ALL" or choose == "ZERO":
        if method =="LINEAR":
            y_pred = model.predict(X_test)
            mdl_rmse = mean_squared_error(Y_test, y_pred) ** 0.5
            mdl_r2 = r2_score(Y_test, y_pred)

            with open('Linear_summary.txt','w') as fh:
                fh.write(ols.summary().as_text())

            with open('Linear_summary.csv', 'w') as fh:
                fh.write(ols.summary().as_csv())

            joblib.dump(model, model_store +'.pkl')
        else:
            y_pred = model.predict(X_test)
            mdl_rmse = mean_squared_error(Y_test, y_pred) ** 0.5
            mdl_r2 = r2_score(Y_test, y_pred)
            joblib.dump(model, model_store +'.pkl')
        return mdl_rmse, mdl_r2, y_pred
    else:
        print("No search method")
