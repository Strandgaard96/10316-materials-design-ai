import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Prepare data

#hist=plt.hist(energies,bins=100)
lam = 10E-4
sigma = 0.0678
l = 1.3

def find_k0(yp,prod):
    k_0 = (1/(len(yp))*np.transpose(yp).dot(prod))
    return k_0

# Construction of k
def kvec(x,nodes,k_0):
    k = np.zeros(len(nodes))
    for i in range(len(nodes)):
        k[i] = k_0*math.exp(-(np.linalg.norm(x-nodes[i,:]))**2/(2*l**2))
    return k

def construct_C(data):
    C = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            C[i, j] = math.exp(-(np.linalg.norm(data[i,:]-data[j,:])) ** 2 / (2 * l ** 2))
    np.fill_diagonal(C, C.diagonal() + sigma**2)
    return C

def calculate_fit(nodes, x_tofit,yp):
    y_fit = np.zeros(x_tofit.shape[0])
    C = construct_C(nodes)
    prod = np.linalg.inv(C).dot(yp)
    k_0 = find_k0(yp,prod)
    C = k_0*C
    prod = np.linalg.inv(C).dot(yp-np.average(yp))

    for i in range(x_tofit.shape[0]):
        k = kvec(x_tofit[i,:], nodes,k_0)
        y_fit[i] = np.transpose(k).dot(prod)+np.average(yp)
    return y_fit

def costruct_training(syss,elemdict, aniondict):
    x = np.zeros((1, 8))
    energy = np.zeros(1)
    for i,elem in enumerate(syss):
        j = random.randrange(0,100);
        if (j<= 10):
            x = np.vstack([x, elemdict[elem.A_ion] + elemdict[elem.B_ion] + aniondict[elem.anion]])
            energy = np.concatenate((energy, [elem.heat_of_formation_all]), axis=0)
    x = np.delete(x, 0, 0)
    energy = np.delete(energy,0,0)
    return x,energy

def costruct_test2(syss,elemdict, aniondict):
    x = np.zeros((1, 8))
    energy = np.zeros(1)
    for i,elem in enumerate(syss):
        j = random.randrange(0,100);
        if j<= 100:
            x = np.vstack([x, elemdict[elem.A_ion] + elemdict[elem.B_ion] + aniondict[elem.anion]])
            energy = np.concatenate((energy, [elem.heat_of_formation_all]), axis=0)
    x = np.delete(x, 0, 0)
    energy = np.delete(energy,0,0)
    return x,energy

def construct_error(prediction, actual):
    plt.figure()
    error = (abs(actual-prediction))/(abs(1+actual))*100
    plt.plot(error,label = f"Relative errror, l = {l}")
    plt.ylabel("Relative error (\%)")
    plt.xlabel("Instance")
    plt.legend()

    plt.figure()
    plt.plot(np.arange(-1,5),np.arange(-1,5))
    plt.plot(actual,prediction,"*",label = f"l = {l}")
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.legend()
    return error


if __name__ == '__main__':
    xp,energy = costruct_training(syss,elemdict,aniondict)
    x_tofit,energy_all = costruct_test2(syss,elemdict,aniondict)
    y_fit = calculate_fit(xp, x_tofit, energy)
    error = construct_error(y_fit, energy_all)
