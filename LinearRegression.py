def linear_regression(X, y, X_test):
    import numpy as np
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

    y_predicted_train=X@w
    print("y_train_classified is" , np.sign(y_predicted_train))

    y_predicted=X_test@w
    print("y_predicted is\n", y_predicted, "\n")
    print("y_predicted_classified is\n", np.sign(y_predicted), "\n")

    # print("X rank:", np.linalg.matrix_rank(X))
    # result=np.hstack((X,y))
    # print("X|y rank: ", np.linalg.matrix_rank(result))


    return(system, w, sum_of_square, mean_squared_error, y_predicted)
