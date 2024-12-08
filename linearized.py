from flint import arb, ctx
from cohom1 import Eta
from arb_cheby import pickleable_polys, unpickle_polys
import numpy as np
import pickle
import os

arb_zeros = lambda n: np.array([arb('0') for i in range(n)])

class DiffEta(object):
    def __init__(self, eta, dps, delta=None, data=None):
        self.d1 = eta.d1
        self.d2 = eta.d2
        self.alpha = eta.alpha
        self.lam = eta.lam
        self.eta = eta
        self.dps = dps
        self.L = eta.L
        ctx.dps = int(7.3 * dps) + 32
        if delta is None:
            delta = arb('0.1')**(dps // 3)
        self.delta = delta
        if data is None:
            eta_left = Eta(self.d1, self.d2, self.alpha-delta, self.lam, dps=dps)
            eta_right = Eta(self.d1, self.d2, self.alpha+delta, self.lam, dps=dps)
            self.T = eta.eta_pT
            eta_left.cheby(eps=0.001, T=self.T)
            eta_right.cheby(eps=0.001, T=self.T)
            d_eta_l1, d_eta_l2, d_eta_l3 = eta_left.eta_p
            d_eta_r1, d_eta_r2, d_eta_r3 = eta_right.eta_p
            # Compute central differences
            self.d_mu_1 = (d_eta_r1 - d_eta_l1)* (arb('1.0') / (2 * self.delta))
            self.d_mu_2 = (d_eta_r2 - d_eta_l2)* (arb('1.0') / (2 * self.delta))
            self.d_mu_3 = (d_eta_r3 - d_eta_l3)* (arb('1.0') / (2 * self.delta))
            t_right = eta_right.mean_curv_root()
            t_left = eta_left.mean_curv_root()
            self.d_stop = (t_right - t_left) / (2 * delta)
            self.d_final = (eta_right.p(t_right) - eta_left.p(t_left)) / (2 * self.delta)
        else:
            self.T = data['T']
            self.d_mu_1, self.d_mu_2, self.d_mu_3 = unpickle_polys(data['polys'], self.T)
            self.d_stop = data['d_stop']
            self.d_final = data['d_final']
            ctx.dps = int(7.3 * dps) + 32
        
    def l2_bounds(self, k=0, upper=False):
        """Rigorously compute the L2/Hk bounds on eta using the Chebyshev polynomial
        approximation"""
        T_a = self.eta.eta_pT
        mu_1 = self.d_mu_1.integral()
        mu_2 = self.d_mu_2.integral()
        mu_3 = self.d_mu_3.integral()
        hk_eta_1 = 0
        hk_eta_2 = 0
        hk_eta_3 = 0
        for l in range(k):
            hk_eta_1 += (mu_1*mu_1).integral().evaluate([T_a])[0]#.sqrt()
            hk_eta_2 += (mu_2*mu_2).integral().evaluate([T_a])[0]#.sqrt()
            hk_eta_3 += (mu_3*mu_3).integral().evaluate([T_a])[0]#.sqrt()
            mu_1 = mu_1.derivative()
            mu_2 = mu_2.derivative()
            mu_3 = mu_3.derivative()
        hk_eta_1 += (mu_1*mu_1).integral().evaluate([T_a])[0]#.sqrt()
        hk_eta_2 += (mu_2*mu_2).integral().evaluate([T_a])[0]#.sqrt()
        hk_eta_3 += (mu_3*mu_3).integral().evaluate([T_a])[0]#.sqrt()
        hk_eta_1, hk_eta_2, hk_eta_3 = hk_eta_1.sqrt(), hk_eta_2.sqrt(), hk_eta_3.sqrt()
        if upper:
            hk_eta_1 = round(float((hk_eta_1 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
            hk_eta_2 = round(float((hk_eta_2 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
            hk_eta_3 = round(float((hk_eta_3 + arb(mid=0, rad=0.005)).upper()), ndigits=2)
        return hk_eta_1, hk_eta_2, hk_eta_3

    def a_posteriori_error(self, k=0, upper=False):
        """Rigorously compute the a posteriori error bounds using the Chebyshev polynomial
        approximation. Error is in terms of the Sobolev Hk norm"""
        T_a = self.T
        d1 = self.d1
        d2 = self.d2
        lam = self.lam
        alpha = self.alpha
        d_eta_1, d_eta_2, d_eta_3 = self.eta.eta_p
        eta_1 = d_eta_1.integral()
        eta_2 = d_eta_2.integral()
        eta_3 = d_eta_3.integral()
        mu_1 = self.d_mu_1.integral()
        mu_2 = self.d_mu_2.integral()
        mu_3 = self.d_mu_3.integral()
        mu_2_x = self.d_mu_2.cesaro()
        mu_3_x = self.d_mu_3.cesaro()
        arb_one = arb('1')
        # Applying differential operator
        B_1 = (eta_2 + mu_2*alpha + eta_1*mu_2 + eta_2*mu_1) * (-arb_one / (2*d1))
        B_2 = (eta_2*mu_3 + eta_3*mu_2)*arb('-0.5') 
        B_2 += d1*(eta_1*mu_1 + alpha + eta_1 + alpha*mu_1)
        B_3 = (eta_2*mu_3 + eta_3*mu_2 - eta_3*mu_3) * (arb_one/d2) - (eta_2*mu_2)*(arb_one/d1+arb_one/d2)
        E_1 = self.d_mu_1 - 2 * B_1
        E_2 = self.d_mu_2 + d2*mu_2_x - 2 * B_2
        E_3 = self.d_mu_3 - 2*mu_2_x + 2*mu_3_x - 2 * B_3
        # Integrate
        hk_err_1 = 0
        hk_err_2 = 0
        hk_err_3 = 0
        for l in range(k+1):
            hk_err_1 += (E_1*E_1).integral().evaluate([T_a])[0]#.sqrt()
            hk_err_2 += (E_2*E_2).integral().evaluate([T_a])[0]#.sqrt()
            hk_err_3 += (E_3*E_3).integral().evaluate([T_a])[0]#.sqrt()
            E_1 = E_1.derivative()
            E_2 = E_2.derivative()
            E_3 = E_3.derivative()
        if upper:
            hk_eta_1 = (hk_eta_1 + arb(mid=0, rad=0.005)).upper()
            hk_eta_2 = (hk_eta_2 + arb(mid=0, rad=0.005)).upper()
            hk_eta_3 = (hk_eta_3 + arb(mid=0, rad=0.005)).upper()
        err = hk_err_1 + hk_err_2 + hk_err_3
        return err.sqrt()

    def __call__(self, t):
        return np.array([self.d_mu_1.integral()(t), 
                         self.d_mu_2.integral()(t), 
                         self.d_mu_3.integral()(t)])
    
    def deriv(self, t):
        return np.array([self.d_mu_1(t), 
                         self.d_mu_2(t), 
                         self.d_mu_3(t)])
    

def linearize_solution_O3O10(eta, zeta, dps):
    print('Generate Linearization with Target DPS:', dps)
    print('----------------------------------------')
    fname = 'data/mu_nu_'+str(dps)+'.pkl'
    if not os.path.isfile(fname):
        mu = DiffEta(eta, dps = dps)
        nu = DiffEta(zeta, dps = dps)
        assert mu.delta.mid() == nu.delta.mid()
        with open(fname,'wb') as file:
            pickle.dump({'T_mu': mu.T.str(radius=False), 'T_nu': nu.T.str(radius=False),
                        'delta': mu.delta.str(radius=False),
                        'rho_m': mu.d_stop.mid().str(radius=False),
                        'rho_r': mu.d_stop.rad().str(radius=False),
                        'sig_m': nu.d_stop.mid().str(radius=False),
                        'sig_r': nu.d_stop.rad().str(radius=False),
                        'eta1': [(x.mid().str(radius=False), x.rad().str(radius=False)) \
                                    for x in mu.d_final],
                        'zeta1': [(x.mid().str(radius=False), x.rad().str(radius=False)) \
                                    for x in nu.d_final],
                        'mu':pickleable_polys([mu.d_mu_1, mu.d_mu_2, mu.d_mu_3]),
                        'nu':pickleable_polys([nu.d_mu_1, nu.d_mu_2, nu.d_mu_3])}, file)
    with open(fname, 'rb') as file:
        data = pickle.load(file)
        print('- Load complete')
    data_mu = {'T': eta.eta_pT, 'polys': data['mu'], 
            'd_stop': arb(data['rho_m'],data['rho_r']),
            'd_final': np.array([
                arb(x[0], x[1]) for x in data['eta1']
            ])}
    data_nu = {'T': zeta.eta_pT, 'polys': data['nu'],
            'd_stop': arb(data['sig_m'],data['sig_r']),
            'd_final': np.array([
                arb(x[0], x[1]) for x in data['zeta1']
            ])}
    mu = DiffEta(eta, dps = dps, delta = arb(data['delta']), data = data_mu)
    nu = DiffEta(zeta, dps = dps, delta = arb(data['delta']), data = data_nu)
    return mu, nu