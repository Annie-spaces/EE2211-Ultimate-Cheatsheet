import numpy as np
Y=np.array([[1.8,2.3,4.7,5.2,8.9,9.0,10.3,11.8]]).T   #an example. The numbers are arranged from small to big. It's split into two nodes between 5.2 to 8.9.
print('Y.shape is ',Y.shape)
Y1=Y[0:4]
Y2=Y[4:]
print(f"Y1 is {Y1}\n")
y_mean=np.mean(Y, axis=0)
print(f"y_mean is {y_mean}\n")
y1_mean=np.mean(Y1, axis=0)
print(f"y1_mean is {y1_mean}\n")
y2_mean=np.mean(Y2, axis=0)
print(f"y2_mean is {y2_mean}\n")

y_correct_root=np.array(y_mean*np.ones((len(Y),1)))
print('y_correct_root',y_correct_root)
Y_correct_splitted=np.vstack((y1_mean*np.ones((len(Y1),1)),y2_mean*np.ones((len(Y2),1))))
print("Y_correct_spliitted_root",Y_correct_splitted)


y_difference_square = np.square(Y - y_correct_root)
sum_of_square = sum(y_difference_square)
mean_squared_error = sum_of_square / Y.shape[0]
print("root square error is for root", sum_of_square)
print("root MEAN square error is", mean_squared_error, "\n")


y_difference_square = np.square(Y - Y_correct_splitted)
sum_of_square = sum(y_difference_square)
mean_squared_error = sum_of_square / Y.shape[0]
print("split square error is", sum_of_square)
print("split MEAN square error is", mean_squared_error, "\n")