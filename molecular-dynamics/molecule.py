import numpy as np
from atom import Atom
from verlet import Verlet
from model import LennartJonesModel

class Molecule(list):

    def get_symbols(self):
        """Returns the chemichal symbol of the molecule"""
        s = []
        for atom in self:
            s.append(atom.sym)
        return s

    def get_positions(self):
        """REturns positions of all the atoms in the molecule"""
        positions = np.zeros((len(self),3))
        i = 0
        for atom in self:
            positions[i,:] = atom.pos
            i+=1
        return positions

    def writetofile(self):
        with open('output.xyz', 'w') as file:
            file.write('{}\n'.format(len(self)))
            file.write("This line has a comment\n")
            for atom in self:
                file.write("{:s} {:.2f} {:.2f} {:.2f}\n".format(atom.sym,atom.pos[0],atom.pos[1],atom.pos[2]))

    def set_positions(self):
        model = LennartJonesModel('H',1,2)
        algo = Verlet(model, self)
        v = algo.run(10)
        for atom in self:
            atom.pos =  atom.pos + v

if __name__ == '__main__':
    H1 = Atom('H', np.array([1, 2, 4]))
    H2 = Atom('H', np.array([1, 4, 4]))
    H3 = Atom('H', np.array([4, 5, 7]))
    O = Atom('O',[1,3,4])

    mol = Molecule([H1,H2,H3])
    symbols = mol.get_symbols()
    positions = mol.get_positions()
    mol.writetofile()
    mol.set_positions()


