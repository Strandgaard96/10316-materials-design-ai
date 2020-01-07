import numpy as np

class LennartJonesModel:
    """Modelling interactions"""
    def __init__(self,element,epsilon,sigma):
        """Create the model
        the parameters are model dependant"""

        self.epsilon = epsilon
        self.sigma = sigma
        self.element = element

    def calculate(self,system):
        """Calculate energy and forces
        REturn two values, The first is the enegy,
        the first is the force on all atoms"""

    # Check same type elements
    for elem in system
        if system.sym != self.element:
            raise ValueError("Elements must be equal")

    E = 0
    F = np.zeros(len(system))
    for elem in system:
        for elem2 in system:
            r = elem.pos - elem2.pos
            E += 4*self.epsilon*((self.sigma/r)^12-(self.sigma/r)^6)
            F[i,j] += 4*self.epsilon*(12*(self.epsilon/r)^12-6*(self.epsilon/r)^6)*(r/r^2)




