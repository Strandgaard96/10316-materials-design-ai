import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.optimize
from Coordinate_optimization.gaussian import GaussianFit
from Coordinate_optimization.steepest import grad_rosen, Rosenbrock


def main():
    i = 0
    tol = 0.01
    x0 = np.array([0.5,0.5])
    model = GaussianFit(0, 0.63)

    while True:
        i += 1
        x_train, f_train = model.costruct_training(10)
        res = scipy.optimize.fmin_cg(model.calculate_fit2, x0,args=(x_train, f_train))
        x0 = res
        if i == 50:
            break
        print(i)
        print(res)
        print(Rosenbrock(res))
        print(model.calculate_fit2(res, x_train, f_train))
    return res

if __name__ == '__main__':
    main()
    """
    model = GaussianFit(0, 0.63)
    x_train,f_train = model.costruct_training(10)
    x0 = np.array([0.9, 0.9])
    res = scipy.optimize.fmin_cg(model.calculate_fit2, x0, fprime = grad_rosen, args=(x_train, f_train))
    print(res)
    print(Rosenbrock(res))
    print(model.calculate_fit2(res,x_train,f_train))"""