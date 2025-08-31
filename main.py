from EnterMetrics import EnterMetrics
from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression
from OneHotLinearClassification import onehot_linearclassification
from pearson_correlation import pearson_correlation
import numpy as np
from sklearn.metrics import mean_squared_error
from EnterMetrics import EnterMetrics

X=EnterMetrics("X")
Y=EnterMetrics("Y")
X_test=EnterMetrics("X_test")

# polynomial_regression(X, Y, order=2, X_test=X_test)
# ridge_poly_regression(X, Y, LAMBDA=1, order=i, form='auto', X_test=X_test)

#adding one for linear NOT FOR POLYNOMIAL
X_fitted=np.hstack((np.ones((len(X),1)),X))
X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))
# linear_regression(X_fitted,Y, X_test_fitted)
# ridge_regression(X_fitted,Y,LAMBDA=0.3, X_test=X_test_fitted, form='auto')
# onehot_linearclassification(X_fitted,Y,X_test_fitted)

# pearson_correlation(X,Y)

# print("did you add offset for X if you are using linear regression? and DON'T use offset for polynomial Regression!" )
