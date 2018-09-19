"""Various HNC solvers for different plasma systems. Default unit system is cgs."""

import numpy as np
from scipy.integrate import trapz
from scipy.special import erfc, erf
from potentials import get_potential

kb = 1.3807e-16
qe = 4.8023e-10
me = 9.1094e-28
mp = 1.6726e-24
h = 6.6261e-27
hbar = 1.0546e-27
a0 = 5.2918e-9


class HNCSolver(object):

    def __init__(self):
        super(HNCSolver, self).__init__()


    def get_therm_wavelength(self, m1, m2, T):
        """Returns the thermal DeBroglie wavelength of two particles.

        Parameters
        ----------
        m1 : float
            The mass of particle 1 in grams.
        m2 : float
            The mass of particle 2 in grams.
        T : float
            Temperature of the system in Kelvin.

        Returns
        -------
        l : float
            Thermal DeBroglie wavelength.

        Notes
        -----
        Throughout the project, the definition of thermal wavelength is
        .. math:: \lambda_{\eta\nu} = \sqrt{\frac{2\pi \hbar^{2}}{\mu_{\eta\nu}k_{b}T}}

        where :math:`\mu_{\eta\nu}` is the reduced mass of the two particles."""
        l = (1./m1) + (1./m2)
        l = 2*np.pi*hbar*hbar*l/(kb*T)
        l = np.sqrt(l)
        return l

    def get_therm_wavelengths(self, ms, T):
        """Get the thermal wavelengths associated with a list of particle masses,
        each of which is ostensibly associated with a unique species.

        Parameters
        ----------
        ms : list of float
            The mass of each particle/species in grams.
        T : float
            Temperature of the system in Kelvin.

        Returns
        -------
        ls : list of float
            Thermal wavelengths for each pair of particles.
        """
        ls = []
        for i in range(len(ms)):
            for j in range(i, len(ms)):
                ls.append(self.get_therm_wavelength(ms[i], ms[j], T))
        return ls


    def get_ion_radius(self, ndens):
        """Get the ion-radius associated with a mixture of electrons and arbitrary
        ions.

        This function returns an opinion of the wigner seitz radius given by
        assuming a is the radius given by the total ion density. For this reason,
        it is assumed the first entry in ndens is the electron density.


        Parameters
        ----------
        ndens : array of float
            The list of number densities of each particle species, the first entry
            should be the electron density.

        Returns
        -------
        a : float
            The ion radius."""
        if len(ndens) == 1:
            a = np.power(3./(4*np.pi*ndens[0]), 1./3)
        else:
            a = np.power(3./(4*np.pi*np.sum(ndens[1:])), 1./3)
        return a


    def get_coupling_params(self, qs, ndens, T):
        """Returns the coupling parameter :math:`\Gamma` for each unique pairwise
        interaction.

        Parameters
        ----------
        qs : list of int
            The charge of each particle species in units of fundamental charge. The
            first entry should be the electron charge.
        ndens : array of float
            A list of number densities of each particle species, the first entry
            should be the electron density.
        T : float
            The temperature of the entire system in Kelvin.

        Returns
        -------
        gammas : list of float
        The pairwise unique coupling parameters.
        """
        gammas = []
        a = self.get_ion_radius(ndens)
        gamma = qe*qe/(a*kb*T)
        for i in range(len(qs)):
            for j in range(i, len(qs)):
                gammas.append(qs[i]*qs[j]*gamma)
        return gammas

    def l2_norm(self, v1, v2, rbar):
        out = v1 - v2
        out = out**2
        out = trapz(out, rbar)
        out = np.sqrt(out)
        return out



class TrapSolver(HNCSolver):

    def __init__(self, res, rbar_lims, rtol, kmin):
        super(TrapSolver, self).__init__()
        self.res = res
        self.rbar = np.linspace(rbar_lims[0], rbar_lims[1], num=res)
        if kmin is None:
            self.k = self.get_k()
        else:
            self.k = self.get_k(kmin)
        self.tol = rtol
        self.error = 1


    def get_k(self, kmin=1e-3):
        """Get the sampled wavelengths for a spherically symmetric fourier transform
        generated with trapezoidal integration.

        Parameters
        ----------
        kmin : float, optional
            The smallest wavelength to sample.

        Returns
        -------
        k : array of float
            Wavelengths to sample.

        Notes
        -----
        As this function will be used in the hnc solvers, it should be noted that
        setting kmin to be too small can lead to singularity issues."""
        k = np.zeros(self.rbar.shape)
        k[0] = kmin
        dk = np.pi(self.rbar[-1] - self.rbar[0])
        for i in range(1, k.shape[0]):
            k[i] = k[i-1] + dk
        return k


    def trap_radial_ft(self, f):
        """Generate the radially symmetric fourier transform of a function using
        trapezoidal integration.

        Parameters
        ----------
        f : array of float
            The values of a function at the corresponding values of rbar.

        Returns
        -------
        fhat : array of float
        The transformed function."""
        fhat = np.zeros(f.shape[0])
        dr = (self.rbar[-1] - self.rbar[0])/f.shape[0]
        for i in range(self.k.shape[0]):
            ki = self.k[i]
            y = self.rbar*f*np.sin(ki*self.rbar)
            fhat[i] = 4*np.pi*trapz(y, dx=dr)/ki
        return fhat


    def itrap_radial_ft(self, fhat):
        """Generate the radially symmetric inverse fourier transform of a function
        using trapezoidal integration.

        Parameters
        ----------
        fhat : array of float
            The values of a function at the corresponding values of k.

        Returns
        -------
        f : array of float
            The values of the transformed function at the corresponding values of r."""
        f = np.zeros(fhat.shape[0])
        dk = (self.k[-1] - self.k[0])/self.k.shape[0]
        for i in range(self.rbar.shape[0]):
            ri = self.rbar[i]
            y = self.k*fhat*np.sin(self.k*ri)
            f[i] = trapz(y, dx=dk)/(2*np.pi*np.pi*ri)
        return f


    def vec_trap_radial_ft(self, fs):
        """Generate the radially symmetric fourier transform of a set of functions
        using trapezoidal integration.

        Parameters
        ----------
        fs : array of arrays of float
            The values of functions at the corresponding values of r.
        r : array of float
            The radial values at which a function is sampled.
        k : array of float
            The wavelengths at which the transforms are sampled.

        Returns
        -------
        fhats : array of arrays of float
            The transformed functions."""
        fhats = np.zeros(fs.shape)
        for i in range(fs.shape[0]):
            fhats[i] = self.trap_radial_ft(fs[i])
        return fhats


    def ivec_trap_radial_ft(self, fhats):
        """Generate the radially symmetric inverse fourier transforms of functions
        using trapezoidal integration.

        Parameters
        ----------
        fhats : array of arrays of float
            The values of functions at the corresponding values of k.
        k : array of float
            The wavelengths at which a transform is sampled.
        r : array of float
            The radial values at which the inverses will be sampled.

        Returns
        -------
        fs : array of arrays of float
            The values of the transformed function at the corresponding values of r."""
        fs = np.zeros(fhats.shape)
        for i in range(fhats.shape[0]):
            fs[i] = self.itrap_radial_ft(fhats[i])
        return fs


    def get_ul(self, gamma, alpha):
        """An approximation of the long range component of the coulomb potential from
        [1]_.

        Parameters
        ----------
        gamma : float
            Interaction coupling parameter.
        alpha : float
            Damping factor dictating convergence to standard coulomb potential.
        rbar : array of float
            The values (in units of a) at which the potential is sampled.

        Returns
        -------
        ul : array of float
        Samples of the potential.

        References
        ----------
        .. [1] Ng, Kin‐Chue. “Hypernetted Chain Solutions for the Classical
               One‐component Plasma up to Γ=7000.” The Journal of Chemical Physics
               61, no. 7 (October 1974): 2680–89. https://doi.org/10.1063/1.1682399.
        """
        ul = gamma*erf(alpha*self.rbar)/self.rbar
        return ul


    def get_ulhat(self, gamma, alpha):
        """The radial fourier transform of an approximation of the long range
        component of the coulomb potential from [1]_.

        Parameters
        ----------
        gamma : float
        Interaction coupling parameter.
        alpha : float
        Damping factor dictating convergence to standard coulomb potential.
        k : array of float
        The wavelengths at which the potential transform is sampled.

        Returns
        -------
        ulhat : array of float
        Samples of the potential transform.

        References
        ----------
        .. [1] Ng, Kin‐Chue. “Hypernetted Chain Solutions for the Classical
               One‐component Plasma up to Γ=7000.” The Journal of Chemical Physics
               61, no. 7 (October 1974): 2680–89. https://doi.org/10.1063/1.1682399.
        """
        ulhat = np.exp(-1*self.k*self.k/(4*alpha*alpha))
        ulhat = 4*np.pi*gamma*ulhat/(self.k**2)
        return ulhat


class OCPSolver(TrapSolver):

    def __init__(self, res, rbar_lims, kmin, ndens, q, m, T, ukey, rtol=1e-11, 
        ng_param =1.08):
        super(OCPSolver, self).__init__(res, rbar_lims, rtol, kmin)
        gamma = self.get_coupling_params([q], [ndens], T)
        a = self.get_ion_radius([ndens])
        l = self.get_therm_wavelength(m, m, T)
        ul = self.get_ul(gamma, ng_param)
        self.tol = gamma * self.tol
        self.ndens = ndens*a*a*a
        self.us = get_potential(ukey, gamma, a, l, self.rbar)
        self.us = us - ul
        self.ulhat = self.get_ulhat(gamma, ng_param)
        self.g = np.zeros(self.rbar)
        self.cs = np.zeros(self.rbar)
        self.ns = np.zeros(self.rbar)
        self.solve()


    def step(self, picard_param):
        self.cs_old = np.copy(self.cs)
        self.cs = self.trap_radial_ft(self.cs)
        c = self.c - self.ulhat
        self.ns = (self.ndens*self.cs*c - self.ulhat)/(1 - self.ndens*c)
        self.ns = self.itrap_radial_ft(self.ns)
        self.g = np.exp(ns - us)
        self.cs = self.g - 1.0 - self.ns
        self.cs = picard_param*self.cs + (1 - picard_param)*self.cs_old
        self.error = self.l2_norm(self.cs, self.cs_old, self.rbar)

    def solve(self, pparam=0.85):
        steps = 0
        while self.error > self.tol:
            steps += 1
            self.step(pparam)
        print("Converged to acceptable error with %d steps.\n" % steps)



class MCPSolver(TrapSolver):

    def __init__(self, res, rbar_lims, kmin, ndens, qs, ms, T, ukeys, rtol=1e-11, ng_param=1.08):
        super(MCPSolver, self).__init__(res, rbar_lims, rtol, kmin)
        gammas = self.get_coupling_params(qs, ndens, T)
        a = self.get_ion_radius(ndens)
        ls = self.get_therm_wavelengths(ms, T)
        self.tol = np.abs(self.gammas)*self.tol
        self.ndens = np.array(ndens)*a*a*a
        self.us_arr = np.zeros((len(gammas), len(self.rbar)))
        self.ulhat_arr = np.zeros(self.us_arr.shape)
        ul = np.zeros(self.us_arr.shape[1])
        for i in range(self.ulhat_arr.shape[0]):
            self.ulhat_arr[i, :] = self.get_ulhat(gamma[i], ng_param)
            ul = self.get_ul(gamma[i], ng_param)
            self.us_arr[i, :] = get_potential(ukeys[i], gamma[i], a, ls[i], self.rbar)
            self.us_arr[i, :] = self.us_arr[i, :] - ul
        self.error = np.ones(len(gammas))
        self.g_arr = np.zeros(self.us_arr.shape)
        self.cs_arr = np.zeros(self.us_arr.shape)
        self.ns_arr = np.zeros(self.us_arr.shape)
        self.cs_arr_old = np.zeros(self.us_arr.shape)
        self.solve()


    def get_symvec_ind(self, i, j, n):
        if i > j:
            k = i
            i = j
            j = k
        out = int(j + (n*i) - ((i*(i+1))/2))
        return out


    def get_hnc_mat(self, chat_vec):
        nspecies = len(self.ndens)
        A = np.zeros((len(chat_vec), len(chat_vec)))
        for i in range(nspecies):
            for j in range(i, nspecies):
                ind = self.get_symvec_ind(i, j, nspecies)
                cind = self.get_symvec_ind(i, i, nspecies)
                A[ind, ind] = 1 - self.ndens[i]*chat_vec[cind]
                for m in range(nspecies):
                    if m != i:
                        chat = chat_vec[self.get_symvec_ind(i, m, nspecies)]
                        ind2 = get_symvec_ind(m, j, nspecies)
                        A[ind, ind2] = -1*self.ndens[m]*chat
        return A


    def get_hhats(self, chat_arr):
        hhat_arr = np.zeros(chat_arr.shape)
        for i in range(chat_arr.shape[1]):
            A = self.get_hnc_mat(chat_arr[:, i])
            hhat_arr[:, i] = np.linalg.solve(A, chat_arr[:, i])
        return hhat_arr



    def get_nshats(self, cshat_arr, ulhat_arr):
        n = len(self.ndens)
        chat_arr = chat_arr - ulhat_arr
        hhat_arr = self.get_hhats(chat_arr)
        nshat_arr = np.zeros(chat_arr.shape)
        for i in range(nshat_arr.shape[1]):
            for j in range(n):
                for k in range(j, nspecies):
                    ns_ind = self.get_symvec_ind(j, k, n)
                    nshat_arr[ns_ind, i] = -1*ulhat_arr[ns_ind, i]
                    for m in range(n):
                        cind = get_symvec_ind(j, m, n)
                        hind  = get_symvec_ind(m, k, n)
                        nshat_arr[ns_ind, i] += self.ndens[m]*chat_arr[cind, i]*hhat_arr[hind, i]
        return nshat_arr


    def step(self, picard_param):
        self.cs_arr_old = np.copy(self.cs_arr)
        self.cs_arr = self.vec_trap_radial_ft(self.cs_arr)
        self.ns_arr = self.get_nshats(self.cs_arr, self.ulhat_arr)
        self.ns_arr = self.ivec_trap_radial_ft(self.ns_arr)
        self.g_arr = np.exp(self.ns_arr - self.us_arr)
        self.cs_arr = self.g_arr - 1.0 - self.ns_arr
        self.cs_arr = picard_param*self.cs_arr + (1 - picard_param)*cs_old
        for i in range(len(self.error)):
            self.error[i] = self.l2_norm(self.cs_arr[i], self.cs_arr_old[i], rbar)

    
    def solve(self, pparam=0.85):
        steps = 0
        cond = True
        for i in range(len(self.error)):
            cond = cond and self.error[i] > self.tol[i]
        while cond:
            steps += 1
            self.step(pparam)
            cond = True
            for i in range(len(self.error)):
                cond = cond and self.error[i] > self.tol[i]
        print("Converged to acceptable error with %d steps.\n" % steps)    