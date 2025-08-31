from numpy.ma.core import identity


def ridge_regression(X, y, LAMBDA, X_test, form="auto"):
    import numpy as np
    if form=="auto":
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")

    if form=="primal form":
        I = np.identity(X.shape[1])
        w = np.linalg.inv(X.T @ X+LAMBDA*I) @ X.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = X.T @ np.linalg.inv(X @ X.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(X) @ y

    print("w is: ")
    print(w)
    print("")

    y_calculated=X@w
    print("y calculated is: \n", y_calculated, "\n")
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    print("y_predicted is\n", y_predicted, "\n")
