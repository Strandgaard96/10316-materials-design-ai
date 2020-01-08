import numpy as np

class Verlet:
    """Class for calculating the velocities"""

    def __init__(self, model, system):
        self.model = model
        self.system = system
        self.v = 0
        self.m = 1

    def run(self,N):
        t = np.arange(N)
        with open('log.txt', 'w') as file:
            for i in t:
                F,E = self.model.calculate(self.system)
                self.v += (1/2)*(1/self.m)*F
                file.write("The Kinetic energy is {0}\n".format(1/2*self.m*self.v**2))
        return self.v

