import numpy as np
from skhippr.odes.AbstractODE import AbstractODE

class ODE12_2(AbstractODE):
    
    def __init__(self, p1, p2, p3, x0, autonomous=True):
        super().__init__(n_dof=2, autonomous=autonomous)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.x = np.asarray(x0, dtype=float)

    def dynamics(self,t=None,x=None):
        
        if x is None:
            x = self.x
            
        x = np.asarray(x)
        if x.ndim == 1:
            u1, u2 = x
            f1 = -u1 + self.p1*(1.0-u1)*np.exp(u2)
            f2 = -u2 + self.p1*self.p2*(1.0-u1)*np.exp(u2) - self.p3*u2
            return np.array([f1,f2])
        else:
            u1 = x[0, :]
            u2 = x[1, :]
            f1 = -u1 + self.p1*(1.0-u1)*np.exp(u2)
            f2 = -u2 + self.p1*self.p2*(1.0-u1)*np.exp(u2) - self.p3*u2
            return np.vstack((f1,f2))
    
    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x
        x = np.asarray(x)
        if x.ndim==1:
            u1,u2=x
            e = np.exp(u2)
            df11 = -1.0 - self.p1*e
            df12 = self.p1*(1.0-u1)*e
            df21 = -self.p1*self.p2*e
            df22 = -1.0 + self.p1*self.p2*(1.0 - u1)*e - self.p3
            J = np.asarray([[df11, df12], 
                            [df21,df22]
            ])
            return J
        else:
            u1 = x[0, :]
            u2 = x[1, :]
            e = np.exp(u2)
            df11 = -1.0 - self.p1*e
            df12 = self.p1*(1.0-u1)*e
            df21 = -self.p1*self.p2*e
            df22 = -1.0 + self.p1*self.p2*(1.0 - u1)*e - self.p3
            J = np.zeros((2,2, x.shape[1]))
            J[0, 0, :] = df11
            J[0, 1, :] = df12
            J[1, 0, :] = df21
            J[1, 1, :] = df22
            return J
        