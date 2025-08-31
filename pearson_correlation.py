def pearson_correlation(X,Y):
    import numpy as np
    meanX = np.mean(X, axis=1)
    standard_devX = np.std(X, axis=1)
    Var_X = np.var(X, axis=1)
    print('meanX',meanX)
    print('stdX',standard_devX)
    print('varX',Var_X)

    print(Y)
    meanY = np.mean(Y, axis=1)
    standard_devY = np.std(Y, axis=1)
    Var_Y = np.var(Y, axis=1)
    print('meanY',meanY)
    print('stdY',standard_devY)
    print('varY',Var_Y)

    pearson = []
    covs = []
    for i in range(len(meanX)):
        cov_sum = 0
        for j in range(6):
            cov_sum += (X[i][j] - meanX[i]) * (Y[0][j] - meanY[0])
        pearson.append(cov_sum / 6 / standard_devY / standard_devX[i])
    print(pearson)
