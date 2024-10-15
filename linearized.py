from flint import arb, arb_mat
from arb_cheby import chebyroots, arb_chebyT
import numpy as np
from tqdm import trange
import time

class ChebyLinsolve(object):

    def __init__(self, eta, eta0):
        self.L = eta.L
        self.eta, self.d1, self.d2 = eta, eta.d1, eta.d2
        self.N = eta.eta_p[0].degree()
        self.a, self.b = arb('0'), eta.eta_pT
        self.eta0 = eta0
        # Construct identity matrix
        self.I = np.array([[arb('0') for i in range(5)] for j in range(5)])
        for k in range(5):
            self.I[k,k] = arb('1')
        print('Constructing problem matrices...')
        M = arb_mat(self._M().tolist())
        v = arb_mat(self._v().tolist())
        print('Solving linear problem...',flush=True)
        time.sleep(.2) # Give time to flush print statement
        coeffs = M.solve(v)
        self.polys = self.construct_polys(coeffs)

    def __call__(self, t):
        return np.array([[self.polys[i](t)] for i in range(3)]+[[self.eta0[3]],[self.eta0[4]]])
    
    def deriv(self, t, k=1):
        polys = [self.polys[i] for i in range(3)]
        for j in range(k):
            for i in range(3):
                polys[i] = polys[i].derivative()
        return np.array([[polys[i](t)] for i in range(3)]+[[arb('0')],[arb('0')]])
    
    def hk_norm(self, k=1, start=0):
        polys = [self.polys[i] for i in range(3)]
        hk = [arb('0'), arb('0'), arb('0')]
        for j in range(k+1):
            for i in range(3):
                if j >= start:
                    hk[i] += (polys[i]*polys[i]).integral()(self.eta.eta_pT)
                polys[i] = polys[i].derivative()
        hk = np.array(hk)
        return hk**0.5
    
    def a_posteriori_error(self, t):
        nu = self(t)
        rhs = self.L @ nu / t + 2*self._B(t) @ nu
        lhs = self.deriv(t)
        return lhs - rhs

    def _B(self, t):
        """Construct matrix representation of the bilinear form."""
        eta, d1, d2 = self.eta.p5(t), self.d1, self.d2
        B = np.array([[arb('0') for i in range(5)] for j in range(5)])
        B[0,0] = -1/(2*d1) * eta[1]
        B[0,1] = -1/(2*d1) * (eta[0] + eta[3])
        B[0,3] = -1/(2*d1) * eta[1]
        B[1,0] = d1*(eta[0] + eta[3])
        B[1,1] = -eta[2] / 2
        B[1,2] = -eta[1] / 2
        B[1,3] = d1*(eta[0] + eta[3])
        B[1,4] = -d1*eta[4]
        B[2,1] = 1/d2 * eta[2] - (1/d1+1/d2)*eta[1]
        B[2,2] = -1/d2 * eta[2] + 1/d2*eta[1]
        B[2,4] = -eta[4]
        return B
    
    def _M(self):
        """Construct operator matrix to find coefficients"""
        a, b, N = self.a, self.b, self.N
        grid = chebyroots(a, b, N)
        M = np.array([[arb('0') for i in range(5*(N+1))] for j in range(5*(N+1))])
        # Generate Chebyshev polynomials
        T_polys = [arb_chebyT([0]*k + [1], a, b) for k in range(N+1)]
        dT_polys = [T.derivative() for T in T_polys]
        # Evaluate polynomials at each Chebyshev node
        T = np.array([[T_polys[j](arb('0')) for j in range(N+1)]] + \
                     [[T_polys[j](grid[i]) for j in range(N+1)] for i in range(N)])
        dT = np.array([[arb('0') for i in range(N+1)]] + \
                     [[dT_polys[j](grid[i]) for j in range(N+1)] for i in range(N)])
        # Compute B matrix at each Chebyshev node
        B_grid = []
        for i in trange(N):
            B_grid.append(2*self._B(grid[i]))
        # Compute the top row of the matrix
        for j in range(0, N+1):
            M[0:5, 5*j:5*(j+1)] = T[0,j] * self.I
        # Compute the remaining rows
        for i in range(1, N+1):
            for j in range(0, N+1):
                M[5*i:5*(i+1), 5*j:5*(j+1)] = \
                    grid[i-1] * dT[i,j] * self.I - T[i,j] * self.L - grid[i-1]*T[i,j]*B_grid[i-1]
        return M
    
    def _v(self):
        """Construct right-hand-side of matrix equation to find coefficients"""
        v = np.array([[arb('0')] for j in range(5*(self.N+1))])
        for i in range(5):
            v[i,:] = self.eta0[i]
        return v
    
    def construct_polys(self, coeffs):
        polys = []
        for idx in range(3):
            j = idx
            poly_c = []
            while j < coeffs.nrows():
                poly_c.append(coeffs[j,0])
                j += 5
            polys.append(arb_chebyT(poly_c, self.a, self.b))
        return polys
    



class GaussODEKernel(object):
    def __init__(self, sigma, L):
        """L is a linear operator representing the ODE:
        
        L(dy_dt, y, t) = 0.
        """
        self.sigma = sigma
        self.sigma2 = self.sigma**2
        self.L = L
    
    def K(self, x1, x2):
        z = (x1 - x2) / self.sigma
        return (-z**2 / 2).exp()
    
    def dK_dx(self, x1, x2):
        """Derivative of kernel with respect to first argument"""
        z = (x1 - x2) / self.sigma
        return -z * (-z**2 / 2).exp() / self.sigma
    
    def d2K_dx(self, x1, x2):
        """Derivative of kernel with respect to both arguments"""
        z = (x1 - x2) / self.sigma
        return (1 - z**2) * (-z**2 / 2).exp() / self.sigma2
    
    def K_L(self, x1, x2):
        K = self.K(x1, x2)
        dK_dx1 = self.dK_dx(x1, x2)
        return self.L(dK_dx1, K, x1)

    def K_LL(self, x1, x2):
        K = self.K(x1, x2)
        dK_dx2 = -self.dK_dx(x1, x2)
        d2K = self.d2K_dx(x1, x2)
        arg1 = self.L(d2K, dK_dx2, x2)
        arg2 = self.L(dK_dx2, K, x2)
        return self.L(arg1, arg2, x1)
    
    def gram(self, abscissas):
        """Gram matrix"""
        N = len(abscissas)
        # Allocate matrix of size (N+1) x (N+1)
        G = [[0]*(N+1)]*(N+1)
        G[0][0] = self.K(arb('0'), arb('0'))
        for i, x in enumerate(abscissas):
            G[0][i] = self.K_L(x, arb('0'))
            G[i][0] = G[0][i]
        for i, x1 in enumerate(abscissas):
            for j, x2 in enumerate(abscissas[:i+1]):
                G[i+1][j+1] = self.K_LL(x1, x2)
                G[j+1][i+1] = G[i+1][j+1]
        return arb_mat(G)


