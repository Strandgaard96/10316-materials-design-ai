import numpy as np
from atom import Atom


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
        for elem in system:
            if elem.sym != self.element:
                raise ValueError("Elements must be equal")

        E = 0
        F = np.zeros(len(system))
        force = 0
        for i,elem in enumerate(system):
            for j,elem2 in enumerate(system):
                if j != i:
                    r = np.linalg.norm(elem.pos - elem2.pos)
                    E += 4*self.epsilon*((self.sigma/r)**12-(self.sigma/r)**6)
                    force += 4*self.epsilon*(12*(self.epsilon/r)**12-6*(self.epsilon/r)**6)*(r/r**2)
            F[i] = force
            force = 0
        return F,E

if __name__ == '__main__':
    H1 = Atom('H',np.array([1,2,4]))
    H2 = Atom('H',np.array([1,4,4]))
    H3 = Atom('H', np.array([4, 5, 7]))
    O = Atom('O',[1,3,4])

    ljmodel = LennartJonesModel('H',1,2)
    F,E = ljmodel.calculate([H1,H2,H3])




