import numpy as np

class Atom:
    """Representing an atom"""
    def __init__(self, chemsym, pos):
        self.sym = chemsym
        self.pos = pos

    def __repr__(self):
        "Magic method printing the vector"
        return "Atom({},{:.3f},{:.3f},{:.3f})".format(self.sym,self.pos[0], self.pos[1], self.pos[2])

if __name__ == '__main__':
    a = Atom('Au',np.array([1,2,4]))
    print(a)
