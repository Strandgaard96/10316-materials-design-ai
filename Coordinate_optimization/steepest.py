import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import random

delta = 0.01
x = np.arange(-2.0,2.0, delta)
y = np.arange(-1.0,3.0, delta)
X,Y = np.meshgrid(x,y)
Z = np.log((2*(Y-X**2))**2+(1-X)**2+1)
levels = np.arange(0.0,10.0,0.5)
fig, ax = plt.subplots()
CS = ax.contour(X,Y,Z,18)
ax.set_title("Adsorption  potetial energy surface")
plt.show()

"""Exercise 1
The ground state structure seems to be (1,1)"""

"""Exercise 2
"""

def Rosenbrock(x):
    return np.log((2*(x[1]-x[0]**2))**2+(1-x[0])**2+1)

def grad_rosen(x):
    return np.array([(-16*x[0]*(x[1]-x[0]**2)-2*(1-x[0]))/\
                     (4*(x[1]-x[0]**2)**2+(1-x[0])**2+1),
                     (8*(x[1]-x[0]**2))/(4*(x[1]-x[0]**2)**2+(1-x[0])**2+1)])

def steepest(fun,fun2,x0):
    converged = False
    xk = x0
    it = 0
    f = Rosenbrock(x0)
    df = grad_rosen(x0)
    tol = 0.001
    maxit = 1000

    while not converged and (it < maxit):
        it = it + 1
        p = -df

        alpha, fc, gc, f, of_prev_prev, df = scipy.optimize.line_search(fun, fun2, xk, p)
        if alpha == None:
            alpha = 0.05

        xk = xk + alpha*p
        f = Rosenbrock(xk)
        df = grad_rosen(xk)

        converged = (np.linalg.norm(df) <= tol)
    return xk,converged,it
if __name__ == '__main__':
    it = np.zeros(10)
    for i in range(10):
        x1 = np.arange(-2,-1,0.1)
        x2 = np.arange(2,3,0.1)
        x0 = np.array([x1[i],x2[i]])
        x, converged,it[i]= steepest(Rosenbrock,grad_rosen, x0)
        print(f"Starting point{x0}")
        print(f"Found minimum{x}")
        print(converged)
        print(f"Number of iterations {it[i]}")
        print(f"Average number of iterations = {sum(it)/10}\n")