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


