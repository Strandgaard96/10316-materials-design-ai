import numpy as np
import matplotlib.pyplot as plt
import math
import ase.db
import random

# Import the relevant data

dbase = ase.db.connect('cubic_perovskites.db')
# Do not include the so-called "reference systems"
syss=[c for c in dbase.select() if not hasattr(c,"reference")]

print(syss[0]._keys)

# Extract information
Aset=set()
Bset=set()
anionset=set()
for p in syss:
    Aset.add(p.A_ion)
    Bset.add(p.B_ion)
    anionset.add(p.anion)
Alist=list(Aset)
Alist.sort()
Blist=list(Bset)
Blist.sort()
anionlist=list(anionset)
anionlist.sort()

# Anions: O,N,S,F
aniondict={'N3':[0,3,0,0],'O2F':[2,0,0,1], 'O2N':[2,1,0,0], 'O2S':[2,0,1,0],'O3':[3,0,0,0], 'OFN':[1,1,0,1],'ON2':[1,2,0,0]}

elemdict={'Ag':[5,11],
          'Al':[3,13],
          'As':[4,15],
          'Au':[6,11],
          'B':[2,13],
          'Ba':[6,2],
          'Be':[2,2],
          'Bi':[6,15],
          'Ca':[4,2],
          'Cd':[5,12],
          'Co':[4,9],
          'Cr':[4,6],
          'Cs':[6,1],
          'Cu':[4,11],
          'Fe':[4,8],
          'Ga':[4,13],
          'Ge':[4,14],
          'Hf':[6,4],
          'Hg':[6,12],
          'In':[5,13],
          'Ir':[6,9],
          'K':[4,1],
          'La':[6,2.5],
          'Li':[2,1],
          'Mg':[3,2],
          'Mn':[4,7],
          'Mo':[5,6],
          'Na':[3,1],
          'Nb':[5,5],
          'Ni':[4,10],
          'Os':[6,8],
          'Pb':[6,14],
          'Pd':[5,10],
          'Pt':[6,10],
          'Rb':[5,1],
          'Re':[6,7],
          'Rh':[5,9],
          'Ru':[5,8],
          'Sb':[5,15],
          'Sc':[4,3],
          'Si':[3,14],
          'Sn':[5,14],
          'Sr':[5,2],
          'Ta':[6,5],
          'Te':[5,16],
          'Ti':[4,4],
          'Tl':[6,13],
          'V':[4,5],
          'W':[6,6],
          'Y':[5,3],
          'Zn':[4,12],
          'Zr':[5,4]}

energies = [s.heat_of_formation_all for s in syss]

hist=plt.hist(energies,bins=100)

# Hyperparameters
lam = 10E-4
l = 1
sigma = 0
k_0 = 1

# Construction of k
def kvec(x,nodes):
    k = np.zeros(len(nodes))
    for i in range(len(nodes)):
        k[i] = k_0*math.exp(-(np.linalg.norm(x-nodes[i,:]))**2/(2*l**2))
    return k

def construct_C(data):
    C = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            C[i, j] = k_0*math.exp(-(np.linalg.norm(data[i,:]-data[j,:])) ** 2 / (2 * l ** 2))
    np.fill_diagonal(C, C.diagonal() + sigma**2)
    return C

def calculate_fit(nodes, x_tofit,yp):
    y_fit = np.zeros(len(nodes))
    C = construct_C(nodes)

    for i in range(x_tofit.shape[0]):
        k = kvec(x_tofit[i,:], nodes)
        y_fit[i] = np.transpose(k).dot(np.linalg.inv(C).dot(yp))
        return y_fit

def costruct_training(syss,elemdict, aniondict):
    x = np.empty(8)
    energy = np.zeros(20000)
    for i,elem in enumerate(syss):
        j = random.randrange(0,100);
        if j< 3:
            x = np.vstack([x,elemdict[elem.A_ion] + elemdict[elem.B_ion] + aniondict[elem.anion]])
            energy[i] = elem.heat_of_formation_all
    x = np.delete(x, 0, 0)
    energy  = energy[energy != 0]
    return x,energy

def costruct_test():
    x_tofit = np.array([[1,2,3,4,7,2,1,8],[1,2,3,4,5,6,7,8]])
    return x_tofit


if __name__ == '__main__':
    xp,energy = costruct_training(syss,elemdict,aniondict)
    x_tofit = costruct_test()
    y_fit = calculate_fit(xp, x_tofit, energy)
