from skhippr.equations.AbstractEquation import AbstractEquation
import numpy as np

class Equilibrium12_2(AbstractEquation):
    def __init__(self, p1,p2,p3,u1s,u2s):
        super().__init__(None)
        self.p1, self.p2, self.p3 = p1,p2,p3
        self.u1s = np.atleast_1d(np.array(u1s, dtype=float))
        self.u2s = np.atleast_1d(np.array(u2s, dtype=float))
        
    def residual_function(self, update=False):
        u1 = np.atleast_1d(self.u1s)[0]
        u2 = np.atleast_1d(self.u2s)[0]
        f1 = -u1 +self.p1*(1.0 - u1)*np.exp(u2)
        f2 = -u2 +self.p1*self.p2*(1.0-u1)*np.exp(u2) - self.p3*u2
        return np.array([f1,f2], dtype=float)