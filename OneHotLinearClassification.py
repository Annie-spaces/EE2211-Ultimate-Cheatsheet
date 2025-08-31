def onehot_linearclassification(X, y, X_test):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    # onehot = OneHotEncoder(sparse_output=False)
    # Y_onehot = onehot.fit_transform(y).toarray()
    # print(Y_onehot)

    #linear regression process
    if X.shape[1]<X.shape[0]:
        system="overdetermined"
    elif X.shape[1]>X.shape[0]:
        system="underdetermined"
    else:
        system="full rank"
    print(system, "system \n")

    if system=="overdetermined":
        w=np.linalg.inv(X.T@X)@X.T@y
    elif system=="underdetermined":
        w=X.T@np.linalg.inv(X@X.T)@y
    else:
        w=np.linalg.inv(X)@y
    print("w is: \n", w, "\n")

    y_calculated=X@w
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    y_predicted=np.argmax(y_predicted,axis=1)
    print("y_predicted is\n", y_predicted, "\n")


