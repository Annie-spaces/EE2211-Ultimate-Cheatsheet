def EnterMetrics(a):
    import numpy as np
    print(a)
    X=[]
    while True:
        row = input()
        if row == "":
            break
        numbers = [float(i) for i in row.split(',')]
        X.append(numbers)
    print(f"{a}=np.array({X})\n")
    X = np.array(X)
    return(X)
