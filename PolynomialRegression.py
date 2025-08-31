def polynomial_regression(X,y,order,X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if P.shape[1] < P.shape[0]:
        system = "overdetermined"
    elif P.shape[1] > P.shape[0]:
        system = "underdetermined"
    else:
        system = "full rank"
    print(system, "system")
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if system == "overdetermined":
        w = np.linalg.inv(P.T @ P) @ P.T @ y
    elif system == "underdetermined":
        w = P.T @ np.linalg.inv(P @ P.T) @ y
    else:
        w = np.linalg.inv(P) @ y
    print("w is: ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is: ", P_train_predicted)
    print("y_train_classified is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_predicted is")
    print(y_predicted)


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #
    # return(system, P, w, y_predicted, y_classified)

    # print("P rank:", np.linalg.matrix_rank(P))
    # result=np.hstack((P,y))
    # print("P|y rank: ", np.linalg.matrix_rank(result))
