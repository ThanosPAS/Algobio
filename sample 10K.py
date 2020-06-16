# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:27:15 2020

@author: white
"""


k=0
# Outer loop
for train_index, test_index in CV1.split(Xr):
    
    # extract training and test set for current CV fold
    Xr_train = Xr[train_index,:]
    yr_train = yr[train_index]
    Xr_test = Xr[test_index,:]
    yr_test = yr[test_index]
    internal_cross_validation = 10
    
    # Convert to tensors
    X_nn_train_fs = torch.Tensor(Xr_train)
    y_nn_train_fs = torch.Tensor(yr_train)
    X_nn_val_fs = torch.Tensor(Xr_test)

    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(yr_train-yr_train.mean()).sum()/yr_train.shape[0]
    Error_test_nofeatures[k] = np.square(yr_test-yr_test.mean()).sum()/yr_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    # Fitting Linear Regression to the dataset
    
    lin_reg = LinearRegression(fit_intercept=True,n_jobs=-1)
    lin_reg.fit(Xr_train, yr_train)
    Error_train[k] = np.square(yr_train-lin_reg.predict(Xr_train)).sum()/yr_train.shape[0]
    Error_test[k] = np.square(yr_test-lin_reg.predict(Xr_test)).sum()/yr_test.shape[0]

    # Compute squared error with feature subset selection
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(Xr_train, yr_train, internal_cross_validation,display=textout)
    #Inner loop
    Features[selected_features,k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        N, M =Xr_train[:,selected_features].shape
        lin_reg = LinearRegression(fit_intercept=True,n_jobs=-1)
        lin_reg.fit(Xr_train[:,selected_features], yr_train)
        Error_train_fs[k] = np.square(yr_train-lin_reg.predict(Xr_train[:,selected_features])).sum()/yr_train.shape[0]
        Error_test_fs[k] = np.square(yr_test-lin_reg.predict(Xr_test[:,selected_features])).sum()/yr_test.shape[0]
   
         #Ridge regularization linear model training based on selected features
        lasso_reg_fs = make_pipeline(PolynomialFeatures(2), Lasso(alpha= 0.3727593720314938))
        lasso_reg_fs.fit(Xr_train[:,selected_features], yr_train)
       
        
        Error_train_fs_lasso[k] = np.square(yr_train-lasso_reg_fs.predict(Xr_train[:,selected_features])).sum()/yr_train.shape[0]
        Error_test_fs_lasso[k] = np.square(yr_test-lasso_reg_fs.predict(Xr_test[:,selected_features])).sum()/yr_test.shape[0]
    
        model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, 3), #M features to n_hidden_units
                                torch.nn.Tanh(), # 1st transfer function,

                                torch.nn.Linear(3, 3),   # torch.nn.ReLU()   torch.nn.Tanh()
                                torch.nn.ReLU(),
            
                                torch.nn.Linear(3, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
        loss_fn = torch.nn.MSELoss()
            
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                                loss_fn,
                                                                X=X_nn_train_fs[:,selected_features],
                                                                y=y_nn_train_fs,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            
        # Determine estimated class labels for test set
        y_nn_val_pred_fs = net(X_nn_val_fs[:,selected_features]).detach().numpy()

        # Calculate error (RMSE)
        nn_error_val_fs[k] = np.sqrt(np.mean((y_nn_val_pred_fs.squeeze()-yr_test)**2))
            
            
            

        # mean_nn_error_val_fs = np.mean(nn_error_val_fs)
        # min_error_nn_val_fs[k] = np.min(mean_nn_error_val_fs)

        # min_error_nn_index_fs = np.where(mean_nn_error_val_fs == min_error_nn_val_fs[k])[0][0]
        # h_opt_val_fs[k] = hidden_units_fs[min_error_nn_index_fs]
        
        
        
    
        figure(k,dpi=300)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), np.sqrt(loss_record[1:]))
        xlabel('Iteration')
        ylabel('RMSE (crossvalidation)')    
        title('Regression model number: {0}'.format(k))
        subplot(1,3,3)
        bmplot(regression_attribute_names, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')
        
    #  # Collect mean of min errors
    # nn_error_val_tot_fs[k] = np.mean(min_error_nn_val_fs)
    # rr_error_val_tot_fs[k] = np.mean(min_error_rr_val_fs)

    print('Cross validation fold {0}/{1}'.format(k+1,K1))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    #-----------------------------------------------------------------------    
    
    
    P, L = Xr.shape
    # Init RMSE
    nn_error_val = np.zeros([K2,len(hidden_units)])
    rr_error_val = np.zeros([K2,len(lambdas)])
    
    # Init optimal lambda & optimal h
    h_opt_val = np.zeros(K2)
    lambda_opt_val = np.zeros(K2)
    
    # Init min error
    min_error_nn_val = np.zeros(K2)
    min_error_rr_val = np.zeros(K2)
    
    
     ##### Inner loop for training and validation #####
    j=0
    for train_index, val_index in CV2.split(Xr_train):
        
        # extract training and test set for current CV fold
        X_train, y_train = Xr_train[train_index,:], yr_train[train_index]
        X_val, y_val = Xr_train[val_index,:], yr_train[val_index]

        # Convert to tensors
        X_nn_train = torch.Tensor(X_train)
        y_nn_train = torch.Tensor(y_train)
        X_nn_val = torch.Tensor(X_val)


        ##### ANN training #####
        for i, h in enumerate(hidden_units):

            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(L, h), #M features to n_hidden_units
                                torch.nn.Tanh(), # 1st transfer function,

                                torch.nn.Linear(h, h),   # torch.nn.ReLU()   torch.nn.Tanh()
                                torch.nn.ReLU(),
            
                                torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss()
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                                loss_fn,
                                                                X=X_nn_train,
                                                                y=y_nn_train,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            
            # Determine estimated class labels for test set
            y_nn_val_pred = net(X_nn_val).detach().numpy()

            # Calculate error (RMSE)
            nn_error_val[j,i] = np.sqrt(np.mean((y_nn_val_pred-y_val)**2))

        # mean_nn_error_val = np.mean(nn_error_val,axis=0) # WRONG
        min_error_nn_val[j] = np.min(nn_error_val[j]) # np.min(mean_nn_error_val)
        min_error_nn_index = np.where(nn_error_val[j] == min_error_nn_val[j])[0][0]
        h_opt_val[j] = hidden_units[min_error_nn_index]
        
        ##### Lasso training #####
        for i, lam in enumerate(lambdas):
            
            lasso_reg_nest = make_pipeline(PolynomialFeatures(2), Lasso(alpha=lam))
            lasso_reg_nest.fit(X_train, y_train)
    
            # Fit ridge regression model
            #ridge_reg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=lam))
            #ridge_reg.fit(X_train, y_train)
    
            # Compute model output:
            y_val_pred = lasso_reg_nest.predict(X_val)
            #y_val_pred = ridge_reg.predict(X_val)
    
            # Calculate error (RMSE)
            rr_error_val[j,i] = np.sqrt(np.mean((y_val_pred-y_val)**2))

        # mean_rr_error_val = np.mean(rr_error_val,axis=0) # WRONG        
        min_error_rr_val[j] = np.min(rr_error_val[j]) # np.min(mean_rr_error_val)
        min_error_rr_index = np.where(rr_error_val[j] == min_error_rr_val[j])[0][0]
        lambda_opt_val[j] = lambdas[min_error_rr_index]
        
        
        
        print('\nK1:',k+1,' K2:',j+1)
        print('min Lasso RMSE error:', np.round(min_error_rr_val[j],4))
        print('min ANN RMSE error:', np.round(min_error_nn_val[j],4))
        print('opt lambda:', np.round(lambdas[min_error_rr_index],4))
        print('opt h:', np.round(hidden_units[min_error_nn_index],4))
    
        j+=1
        
    h_opt[k] = np.round(np.mean(h_opt_val)).astype(int)
    lambda_opt[k] = np.mean(lambda_opt_val)

    # Collect mean of min errors
    nn_error_val_tot[k] = np.mean(min_error_nn_val)
    rr_error_val_tot[k] = np.mean(min_error_rr_val)


    print('\nmean lasso val error', np.round(np.mean(min_error_rr_val),4))
    print('mean ANN val error', np.round(np.mean(min_error_nn_val),4))
    print('mean lambda', np.round(lambda_opt[k],4))
    print('mean h', h_opt[k])
    # print('most frequent h', np.argmax(np.bincount(h_opt_val.astype(int))))



    ##### Validation using test data #####

    # ANN testing    
    X_nn_par = torch.Tensor(X_train)
    y_nn_par = torch.Tensor(y_train)
    X_nn_test = torch.Tensor(X_val)

    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(L, h_opt[k]),
                        torch.nn.Tanh(),
                        torch.nn.Linear(h_opt[k], h_opt[k]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(h_opt[k], 1),
                        )
    loss_fn = torch.nn.MSELoss()

    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_nn_par,
                                                        y=y_nn_par,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
    y_nn_test_pred = net(X_nn_test).detach().numpy()
    nn_error[k] = np.sqrt(np.mean((y_nn_test_pred-yr_test)**2))
        
    
    # Lasso testing
    lasso_reg_val = make_pipeline(PolynomialFeatures(2), Lasso(alpha=lambda_opt[k]))
    lasso_reg_val.fit(Xr_train, yr_train)
    y_test_pred = lasso_reg_val.predict(Xr_test)
    rr_error[k] = np.sqrt(np.mean((y_test_pred-yr_test)**2))


    # Baseline testing
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(Xr_train, yr_train)
    y_bl_pred = lin_reg.predict(Xr_test)
    bl_error[k] = np.sqrt(np.mean((y_bl_pred-yr_test)**2))  # root mean square error
    


    ##### Statistic evaluation #####
    nn_rr.append( np.mean( np.abs( y_nn_test_pred-yr_test ) ** loss - np.abs( y_test_pred-yr_test) ** loss ) )
    nn_bl.append( np.mean( np.abs( y_nn_test_pred-yr_test ) ** loss - np.abs( y_bl_pred-yr_test) ** loss ) )
    rr_bl.append( np.mean( np.abs( y_test_pred-yr_test ) ** loss - np.abs( y_bl_pred-yr_test) ** loss ) )


    k+=1
    