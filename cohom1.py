import numpy as np
from tqdm import trange, tqdm
from flint import arb, arb_poly, ctx, arb_mat
from arb_cheby import pickleable_polys, unpickle_polys, chebyfit
from arb_roots import brent, broyden
from scipy.io import savemat
import sys
import os
import pickle

arb_zeros = lambda n: np.array([arb('0') for i in range(n)])
    
def _check_precision(vals, dps):
    """Check that the values are within the given precision."""
    return all([v.rad().log() < -dps * arb('10').log() for v in vals])
        
class Eta(object):
    """Compute the function $\\eta(t)$ to arbitrary precision. This is conducted in two parts:
    1. An approximate (heuristic) solution is constructed using a Taylor series solver.
    2. A high-degree polynomial is fitted to the output of the approximate solution.
    """

    def __init__(self, d1, d2, alpha, lam, dps=20, rho=0.3, working_dps=None, verbose=False, \
                pverbose = False):
        # Prepare, then set
        self.dps = dps
        if working_dps is None:
            self.working_dps = int(7.3*dps) + 16
        else:
            self.working_dps = working_dps
        ctx.dps = self.working_dps
        # Initialise variables
        self.d1 = arb(str(d1))
        self.d2 = arb(str(d2))
        self.alpha = arb(str(alpha))
        self.lam = arb(str(lam))
        self.verbose = verbose
        self.rho = arb(str(rho))
        self.eta_p = None
        # Set up linearisation matrix
        self.L = np.array([arb_zeros(5) for i in range(5)])
        self.L[1,1] = -self.d2
        self.L[2,1] = arb('2')
        self.L[2,2] = -arb('2')
        # Compute initial power series
        if self.verbose:
            print('ETA(dps = %d)'%self.dps)
            print('-- Series 1')
        t_pt, initial_polys = self.initial_series()
        self.grid = [t_pt]
        self.polys = [initial_polys]
        # Compute remaining series
        if pverbose:
            pbar = tqdm(range(8))
        while self.Z(self.grid[-1]) > 0:
            if self.verbose:
                print('-- Series', len(self.grid)+1)
            t, p = self.next_series()
            self.grid.append(t)
            self.polys.append(p)
            if pverbose:
                pbar.update(1)
                pbar.refresh()
        # Cleanup series (remove error intervals)
        for idx, poly_arr in enumerate(self.polys):
            for idy, poly in enumerate(poly_arr):
                coeffs = poly.coeffs()
                coeffs = [c.mid() for c in coeffs]
                self.polys[idx][idy] = arb_poly(coeffs)


    def __call__(self, t):
        """Evaluate the function at a given time"""
        result = arb_zeros(3)
        for idx, t0 in enumerate(self.grid):
            if t <= t0 + arb('0.00000000001'):
                break
        if idx > 0:
            t_tmp = (t - self.grid[idx-1]) / self.rho
        else:
            t_tmp = t
        result[0] = self.polys[idx][0](t_tmp)
        result[1] = self.polys[idx][1](t_tmp)
        result[2] = self.polys[idx][2](t_tmp)
        return result

    def initial_series(self):
        """Construct the initial power series approximation about zero"""
        t_pt = 0.5 / (12.5 * (self.alpha + self.lam.sqrt()))
        # Compute number of terms required from obtained bound
        N = (self.dps * arb('10').log() + arb('25').log()) / arb('2').log()
        N = int(float(N.mid())) + 1
        # Set initial condition (coefficient zero)
        coeff0 = arb_zeros(5)
        coeff0[3] = self.alpha
        coeff0[4] = self.lam.sqrt()
        coeffs = [coeff0]
        # Recursively construct coefficients
        if self.verbose:
            pbar = trange(N)
        else:
            pbar = range(N)
        for m in pbar:
            coeff = arb_zeros(5)
            for k in range(m + 1):
                coeff += self.B(coeffs[k], coeffs[m-k])
            coeff = self.L_inv(m + 1) @ coeff
            coeffs.append(coeff)
        return t_pt.mid(), [arb_poly([coeffs[idx][idy] for idx in range(N+1)]) for idy in range(3)]

    def next_series(self):
        """Construct the next Taylor series approximation"""
        t_pt = self.grid[-1]
        # Compute first coefficient
        coeff0 = arb_zeros(5)
        coeff0[:3] = self(t_pt)
        assert _check_precision(coeff0[:3], self.dps), 'Not enough working precision!'
        coeff0[0] = coeff0[0].mid()
        coeff0[1] = coeff0[1].mid()
        coeff0[2] = coeff0[2].mid()
        coeff0[3] = self.alpha
        coeff0[4] = self.lam.sqrt()
        coeffs = [coeff0]
        # Compute number of terms required from heuristic
        N = (self.dps + 3)*arb('10').log() / arb('2').log()
        N = int(float(N.mid())) + 1
        # Recursively construct coefficients
        if self.verbose:
            pbar = trange(N)
        else:
            pbar = range(N)
        for m in pbar:
            coeff = arb_zeros(5)
            for k in range(m + 1):
                coeff += self.B(coeffs[k], coeffs[m-k])
                coeff += (-self.rho)**(m-k) / t_pt**(m-k+1) * self.L @ coeffs[k]
            coeff = self.rho / (m + 1) * coeff
            # Stop producing coefficients if they are zero!
            if all([abs(c.mid()) <= c.rad() for c in coeff]):
                N = m
                break
            coeffs.append(coeff)
        # Estimate radius of convergence
        self.radius = self.rho / max([np.max(np.abs(coeffs[N-k]))**(1/(N-k)) for k in range(2,4)])
        assert not np.isnan(float(self.radius.mid())), 'Not enough working precision!'
        return (t_pt+0.5*self.radius).mid(), \
                [arb_poly([coeffs[idx][idy] for idx in range(N+1)]) for idy in range(3)]

    def cheby(self, eps=0.05):
        """Return the Chebyshev polynomial approximations for eta_1, eta_2, eta_3,
        on [0,T+eps] requiring that all three are zero at t = 0."""
        N = int(3 * self.dps) + 1
        ctx.dps = N + 10
        T = self.mean_curv_root().mid() + eps
        if self.verbose:
            print('-- Compute Chebyshev 1')
        eta_p1 = chebyfit(lambda t: self.deriv(t)[0], arb('0'), T, N, verbose=self.verbose)
        if self.verbose:
            print('-- Compute Chebyshev 2')
        eta_p2 = chebyfit(lambda t: self.deriv(t)[1], arb('0'), T, N, verbose=self.verbose)
        if self.verbose:
            print('-- Compute Chebyshev 3')
        eta_p3 = chebyfit(lambda t: self.deriv(t)[2], arb('0'), T, N, verbose=self.verbose)
        self.eta_p = (eta_p1, eta_p2, eta_p3)
        self.eta_pT = T
        return self.eta_p

    def L_inv(self, k):
        """Compute the inverse of the matrix $k I - L$"""
        L_inv = np.array([arb_zeros(5) for i in range(5)])
        L_inv[0,0] = arb('1') / k
        L_inv[1,1] = arb('1') / (k + self.d2)
        L_inv[2,2] = arb('1') / (k + 2)
        L_inv[2,1] = (arb('2') / (k + 2)) / (self.d2 + k)
        L_inv[3,3] = arb('1') / k
        L_inv[4,4] = arb('1') / k
        return L_inv

    def B(self, x, y):
        """Compute the bilinear form $B[x, y]$ in the ODE"""
        res = arb_zeros(5)
        # x3 = y3 = self.alpha
        # x4 = y4 = mpm.sqrt(self.lam)
        res[0] = -(x[1]*y[3]+x[3]*y[1]+x[0]*y[1]+x[1]*y[0])/(2*self.d1)
        res[1] = -(x[1]*y[2] + x[2]*y[1])/2
        res[1] += self.d1*(x[0]*y[0]+x[3]*y[3]+x[0]*y[3]+x[3]*y[0])
        res[1] += -self.d1*x[4]*y[4]
        res[2] = -x[2]*y[2]/self.d2 + (x[1]*y[2]+x[2]*y[1])/self.d2
        res[2] += -(1/self.d1 + 1/self.d2)*x[1]*y[1] - x[4]*y[4]
        return res

    def deriv(self, t, x = None):
        """Compute the derivative of eta --- d/dt eta --- using the ODE"""
        if x is None:
            x = self(t)
        res = arb_zeros(3)
        res[0] = -x[1]*(self.alpha + x[0]) / self.d1
        res[1] = -self.d2*x[1]/t - x[1]*x[2] + self.d1*(self.alpha + x[0])**2 - self.d1*self.lam
        res[2] = 2*x[1]/t - 2*x[2]/t - x[2]**2 / self.d2 - x[1]**2*(1/self.d1 + 1/self.d2)
        res[2] += 2*x[1]*x[2]/self.d2 - self.lam
        return res

    def X(self, t):
        return self.alpha + self(t)[0]

    def Y(self, t):
        return self(t)[1]

    def Z(self, t):
        return self(t)[2] + self.d2 / t

    def W(self, T):
        X = self.X(T)
        Y = self.Y(T)
        Z = self.Z(T)
        numer = (self.d1 + self.d2 - 1)*self.lam - self.d1 * X**2
        numer += (self.d1 - 1)/self.d1 * Y**2
        numer += (self.d2 - 1)/self.d2 * (Y - Z)**2
        numer += 2*Y*(Z - Y)
        return (numer / self.d2).sqrt()
    
    def mean_curv_root(self):
        return brent(self.Z, self.grid[-2], self.grid[-1], self.dps)

    def poly_WXYZ(self, t):
        if self.eta_p is None:
            raise Exception('Need to compute Chebyshev polynomial first!')
        if t > self.eta_pT:
            raise Exception('Cannot evaluate polynomial beyond interpolation interval')
        d_eta_1, d_eta_2, d_eta_3 = self.eta_p
        eta_1 = d_eta_1.integral()
        eta_2 = d_eta_2.integral()
        eta_3 = d_eta_3.integral()
        T_a = t
        e1 = eta_1(T_a)
        e2 = eta_2(T_a)
        e3 = eta_3(T_a)
        d1 = self.d1
        d2 = self.d2
        lam = self.lam
        alpha_a = self.alpha
        X = alpha_a + e1
        Y = e2
        Z = e3 + d2 / T_a
        numer = (d1 + d2 - 1)*lam - d1 * X**2
        numer += (d1 - 1)/d1 * Y**2
        numer += (d2 - 1)/d2 * (Y - Z)**2
        numer += 2*Y*(Z - Y)
        W = (numer / d2).sqrt()
        return {'W': W, 'X': X, 'Y': Y, 'Z': Z, 'eta_1': e1, 'eta_2': e2, 'eta_3': e3}

    def l2_bounds(self, k=0, upper=False):
        """Rigorously compute the L2/Hk bounds on eta using the Chebyshev polynomial
        approximation"""
        if self.eta_p is None:
            raise Exception("Need to compute Chebyshev polynomial first!")
        T_a = self.eta_pT
        eta_1, eta_2, eta_3 = self.eta_p
        eta_1 = eta_1.integral()
        eta_2 = eta_2.integral()
        eta_3 = eta_3.integral()
        hk_eta_1 = 0
        hk_eta_2 = 0
        hk_eta_3 = 0
        for l in range(k+1):
            hk_eta_1 += (eta_1*eta_1).integral().evaluate([T_a])[0].sqrt()
            hk_eta_2 += (eta_2*eta_2).integral().evaluate([T_a])[0].sqrt()
            hk_eta_3 += (eta_3*eta_3).integral().evaluate([T_a])[0].sqrt()
            eta_1 = eta_1.derivative()
            eta_2 = eta_2.derivative()
            eta_3 = eta_3.derivative()
        if upper:
            hk_eta_1 = round(float((hk_eta_1 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
            hk_eta_2 = round(float((hk_eta_2 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
            hk_eta_3 = round(float((hk_eta_3 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
        return hk_eta_1, hk_eta_2, hk_eta_3

    def a_posteriori_error(self, k=0, upper=False):
        """Rigorously compute the a posteriori error bounds using the Chebyshev polynomial
        approximation. Error is in terms of the Sobolev Hk norm"""
        if self.eta_p is None:
            raise Exception('Need to compute Chebyshev polynomial first!')
        # Convert to arb
        T_a = self.eta_pT
        d1 = self.d1
        d2 = self.d2
        lam = self.lam
        alpha_a = self.alpha
        d_eta_1, d_eta_2, d_eta_3 = self.eta_p
        eta_1 = d_eta_1.integral()
        eta_2 = d_eta_2.integral()
        eta_3 = d_eta_3.integral()
        eta_2_x = d_eta_2.cesaro()
        eta_3_x = d_eta_3.cesaro()
        arb_one = arb('1')
        # Applying differential operator
        E_1 = d_eta_1 + eta_2*(alpha_a + eta_1) * (arb_one/d1)
        E_2 = d_eta_2 + d2*eta_2_x + eta_2*eta_3 - d1*(alpha_a+eta_1)*(alpha_a+eta_1) + d1*lam
        E_3 = d_eta_3 - 2*eta_2_x + 2*eta_3_x + eta_3*eta_3 * (arb_one/d2) +\
                eta_2*eta_2 * (arb_one/d1 + arb_one/d2) - 2*eta_2*eta_3*(arb_one/d2) + lam
        # Integrate
        hk_err_1 = 0
        hk_err_2 = 0
        hk_err_3 = 0
        for l in range(k+1):
            hk_err_1 += (E_1*E_1).integral().evaluate([T_a])[0].sqrt()
            hk_err_2 += (E_2*E_2).integral().evaluate([T_a])[0].sqrt()
            hk_err_3 += (E_3*E_3).integral().evaluate([T_a])[0].sqrt()
            E_1 = E_1.derivative()
            E_2 = E_2.derivative()
            E_3 = E_3.derivative()
        if upper:
            hk_eta_1 = (hk_eta_1 + arb(mid=0, rad=0.005)).upper()
            hk_eta_2 = (hk_eta_2 + arb(mid=0, rad=0.005)).upper()
            hk_eta_3 = (hk_eta_3 + arb(mid=0, rad=0.005)).upper()
        err = hk_err_1 + hk_err_2 + hk_err_3
        return err

class AlphaOmega(object):
    """Computing the function $A(t): R -> R^2$ to arbitrary precision using heuristics."""

    def __init__(self, d1, d2, lam, dps=20, working_dps=None, verbose=False):
        # Prepare, then set
        self.dps = dps
        if working_dps is None:
            self.working_dps = int(7.3*(dps+5)) + 16
        else:
            self.working_dps = working_dps
        ctx.dps = self.working_dps
        # Initialise
        self.d1 = d1
        self.d2 = d2
        self.lam = lam
        self.dps = dps
        self.verbose = verbose

    def __call__(self, alpha, forward=True):
        if forward:
            eta = Eta(self.d1, self.d2, alpha, self.lam, dps=self.dps+5, pverbose=self.verbose)
            stop_time_eta = eta.mean_curv_root()
            return eta.W(stop_time_eta), eta.Y(stop_time_eta)
        else:
            zeta = Eta(self.d2, self.d1, alpha, self.lam, dps=self.dps+5, pverbose=self.verbose)
            stop_time_zeta = zeta.mean_curv_root()
            return zeta.X(stop_time_zeta), zeta.Y(stop_time_zeta)
        
    def get_intersection_pt(self, x0):
        """Solve for the intersection point where eta.W = zeta.X and eta.Y = zeta.Y
        using Broyden's method."""
        def F(x):
            alpha, omega = x.entries()
            W1, Y1 = self(alpha, forward=True)
            X2, Y2 = self(omega, forward=False)
            return arb_mat([[W1 - X2], [Y1 - Y2]])
        roots = broyden(F, x0, self.dps).entries()
        if self.verbose:
            print('----------------------------------------')
            print('- alpha_star =', roots[0])
            print('- omega_star =', roots[1])
        return roots

def alpha_posteriori_error(eta, zeta):
    T_alpha = eta.mean_curv_root()
    T_omega = zeta.mean_curv_root()
    error = (eta.poly_WXYZ(T_alpha)['W'] - zeta.poly_WXYZ(T_omega)['X'])**2
    error += (eta.poly_WXYZ(T_alpha)['Y'] - zeta.poly_WXYZ(T_omega)['Y'])**2
    return error.sqrt()

def generate_solution_O3O10(dps, eps=0.005):
    # Avoid issues with conversion to integer
    sys.set_int_max_str_digits(0)
    print('Generate Solution with Target DPS:', dps)
    print('----------------------------------------')
    fname = 'eta_zeta_'+str(dps)+'.pkl'
    if os.path.isfile(fname):
        ctx.dps = int(7.3 * dps) + 16
        with open(fname, 'rb') as file:
            data = pickle.load(file)
            print('- Load complete')
        if data['eps'] != eps:
            raise Exception('Epsilon mismatch, must be recomputed')
        alpha_star, omega_star = arb(data['alpha_star']), arb(data['omega_star'])
        T_alpha, T_omega = arb(data['T_alpha']), arb(data['T_omega'])
        eta = Eta(2, 9, alpha_star, 2+9, dps = dps, verbose=True)
        zeta = Eta(9, 2, omega_star, 2+9, dps = dps, verbose=True)
        eta.eta_pT = T_alpha+eps
        eta.eta_p = unpickle_polys(data['eta'], eta.eta_pT)
        zeta.eta_pT = T_omega+eps
        zeta.eta_p = unpickle_polys(data['zeta'], zeta.eta_pT)
    else:
        ao = AlphaOmega(2, 9, 2+9, dps = dps, verbose=True)
        x0 = arb_mat([[arb('6.0838654955812261377445980424896688618103246957')], \
              [arb('6.1859148331798468647277049839762331689687761753054233564769')]])
        alpha_star, omega_star = ao.get_intersection_pt(x0)
        eta = Eta(2, 9, alpha_star, 2+9, dps = dps, verbose=True)
        T_alpha = eta.mean_curv_root()
        eta.cheby(eps=eps)
        zeta = Eta(9, 2, omega_star, 2+9, dps = dps, verbose=True)
        T_omega = zeta.mean_curv_root()
        zeta.cheby(eps=eps)
        
        with open('eta_zeta_'+str(dps)+'.pkl','wb') as file:
            pickle.dump({'alpha_star':alpha_star.str(radius=False),
                         'omega_star':omega_star.str(radius=False),
                         'T_alpha':T_alpha.str(),'T_omega':T_omega.str(),
                         'eta':pickleable_polys(eta.eta_p),
                         'zeta':pickleable_polys(zeta.eta_p),
                         'eps':eps}, file)
    return eta, zeta, alpha_star, omega_star, T_alpha, T_omega

def pkl_to_matlab(dps, eps=0.005):
    """Export pickled data into MATLAB compatible matrix for use with MATLAB
    scripts."""
    fname = 'eta_zeta_' + str(dps) + '.pkl'
    mname = 'matlab/eta_zeta.mat'
    ctx.dps = int(7.3 * dps) + 16
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    T_alpha, T_omega = arb(data['T_alpha']), arb(data['T_omega'])
    T_alpha += eps
    T_omega += eps
    eta_p = unpickle_polys(data['eta'], T_alpha)
    eta_p = [fn.integral() for fn in eta_p]
    zeta_p = unpickle_polys(data['zeta'], T_omega)
    zeta_p = [fn.integral() for fn in zeta_p]
    M = len(data['eta'])
    N_eta = len(data['eta'][0])
    eta_mat  = np.array([[float(eta_p[idx].coeffs()[idy].mid()) for idy in range(N_eta)] for idx in range(M)])
    zeta_mat = np.array([[float(zeta_p[idx].coeffs()[idy].mid()) for idy in range(N_eta)] for idx in range(M)])
    def a2f(x):
        return float(arb(x).mid())
    savemat(mname, {'eta_mat':eta_mat, 'zeta_mat':zeta_mat, \
                    'T_alpha':float(T_alpha), 'T_omega':float(T_omega), 'eps':float(eps)})
    print('Successfully exported to mat file.')
    