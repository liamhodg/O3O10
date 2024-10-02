from flint import arb, arb_mat


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


