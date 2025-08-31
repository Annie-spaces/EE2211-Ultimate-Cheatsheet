def ridge_poly_regression(X,y,LAMBDA,order, form, X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(P) @ y

    print("w is: ")
    print(w)
    print("")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_predicted is")
    print(y_predicted)

    P_train_predicted=P@w
    print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #HI
    # return(system, P, w, y_predicted, y_classified))










def ridge_poly_regression_simplified(X,y,LAMBDA,order, form, X_test, y_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    # print("the number of parameters: ", P.shape[1])
    # print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    # print(system, "system   ", form)
    # print("")
    # print("the polynomial transformed matrix P is:")
    # print(P)
    # print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(P) @ y

    # print("w is: ")
    # print(w)
    # print("")

    P_test = poly.fit_transform(X_test)
    # print("transformed test sample P_test is")
    # print(P_test)
    # print("")
    y_predicted = P_test @ w
    # print("y_predicted is")
    # print(y_predicted)

    P_train_predicted=P@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    print("ridge train MEAN square error is", mean_squared_error)

    P_test_predicted=P_test@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_test_predicted-y_test)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y_test.shape[0]
    # print("square error is", sum_of_square)
    print("ridge test MEAN square error is", mean_squared_error, "\n")

