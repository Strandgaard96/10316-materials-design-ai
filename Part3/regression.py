import numpy as np
import matplotlib.pyplot as plt
import math
import random


class Fit:
"""Non complete class to fit on the matrix fingerprint"""

    def __init__(self,sigma, l):
        self.lam = 10E-4
        self.sigma = sigma
        self.l = l

    def _find_k0(self, yp,prod):
        k_0 = (1/(len(yp))*np.transpose(yp).dot(prod))
        return k_0

    # Construction of k
    def _kvec(self, x,nodes,k_0):
        k = np.zeros(len(nodes))
        for i in range(len(nodes)):
            k[i] = k_0*math.exp(-(np.linalg.norm(x-nodes[i,:]))**2/(2*self.l**2))
        return k

    def _construct_C(self, data):
        C = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                C[i, j] = math.exp(-(np.linalg.norm(data[i,:]-data[j,:])) ** 2 / (2 * self.l ** 2))
        np.fill_diagonal(C, C.diagonal() + self.sigma**2)
        return C

    def Coulomb(ase_obj):
        pos = ase_obj.get_positions()
        mass = ase_obj.get_atomic_numbers()

        M = np.zeros((len(mass), len(mass)))

        for i in range(len(mass)):
            for j in range(len(mass)):
                Z1 = mass[i]
                Z2 = mass[j]
                if i == j:
                    M[i, j] = 0.5 * Z1 ** 2.4
                else:
                    M[i, j] = (Z1 * Z2) / (np.linalg.norm(pos[i] - pos[j]))
        shap = M.shape
        M = np.sort(M.flatten())
        M = M.reshape(shap)
        return M

    def calculate_fit(self, x_tofit,nodes,yp):
        y_fit = np.zeros(x_tofit.shape[0])
        C = self._construct_C(nodes)
        prod = np.linalg.inv(C).dot(yp)
        k_0 = self._find_k0(yp,prod)
        C = k_0*C
        prod = np.linalg.inv(C).dot(yp-np.average(yp))

        for i in range(x_tofit.shape[0]):
            k = self._kvec(x_tofit[i,:], nodes,k_0)
            y_fit[i] = np.transpose(k).dot(prod)+np.average(yp)
        return y_fit

    def costruct_training(self,n):
        x_train = np.zeros((n,2))
        f_value = np.zeros(n)
        random.seed(10)
        for i in range(n):
            x_train[i,0] = random.uniform(-2,2)
            x_train[i,1] = random.uniform(-1,3)
            f_value[i] = np.log((2*(x_train[i,1]-x_train[i,0]**2))**2+(1-x_train[i,0])**2+1)
        return x_train,f_value


    def costruct_test(self, n):
        x_test = np.zeros((n,2))
        f_test = np.zeros(n)
        random.seed(11)
        for j in range(n):
            x_test[j, 0] = random.uniform(-2, 2)
            x_test[j, 1] = random.uniform(-1, 3)
            f_test[j] = np.log((2 * (x_test[j, 1] - x_test[j, 0] ** 2)) ** 2 + (1 - x_test[j, 0]) ** 2 + 1)
        return x_test, f_test

    def construct_error(self, prediction, actual):
        plt.figure()
        error = (abs(actual-prediction))/(abs(1+actual))*100
        plt.plot(error,label = f"Relative errror, l = {self.l}")
        plt.ylabel("Relative error (\%)")
        plt.xlabel("Instance")
        plt.legend()


        plt.figure()
        f = np.log((2*(x_train[:,1]-x_train[:,0]**2))**2+(1-x_train[:,0])**2+1)
        plt.plot(f,f)
        plt.plot(actual,prediction,"*",label = f"l = {self.l}")
        plt.xlabel("Actual")
        plt.ylabel("Prediction")
        plt.legend()
        return error


if __name__ == '__main__':
    model = Fit(0,0.63)
    x_train, f_train = model.costruct_training(50)
    x_test,f_test = model.costruct_test(1000)
    y_fit = model.calculate_fit2(np.array([-1,2]), x_train, f_train)
    error = model.construct_error(y_fit, f_test)
