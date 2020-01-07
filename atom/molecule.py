import numpy as np
from atom import Atom

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
        with open('output.xyz', 'a') as file:
            file.write('{}\n'.format(len(self)))
            file.write("This line has a comment\n")
            for atom in self:
                file.write("{:s} {:.2f} {:.2f} {:.2f}\n".format(atom.sym,atom.pos[0],atom.pos[1],atom.pos[2]))

if __name__ == '__main__':
    H1 = Atom('H',[1,2,4])
    H2 = Atom('H',[1,4,4])
    O = Atom('O',[1,3,4])

    mol = Molecule([H1,H2,O])
    symbols = mol.get_symbols()
    positions = mol.get_positions()
    mol.writetofile()


