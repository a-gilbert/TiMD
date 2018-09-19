"""Temperature normalized potentials used in solving the hnc system for
   a mixture."""

import numpy as np
from scipy.special import erfc, erf


def coul(gamma, a, l, rbar):
    """The standard coulomb potential.

    Parameters
    ----------
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        The thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The potential at the given sample points."""
    u = gamma/rbar
    return u


def hansen_cd(gamma, a, l, rbar):
    """The hansen qsp [1]_ taking into account quantum diffraction.

    Parameters
    ----------
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        The thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The potential at the given sample points.

    References
    ----------
    .. [1] Hansen, J. P., and I. R. McDonald. “Microscopic Simulation of a
       Strongly Coupled Hydrogen Plasma.” Physical Review A 23, no. 4
       (April 1, 1981): 2041–59. https://doi.org/10.1103/PhysRevA.23.2041.
    """
    u = 1 - np.exp(-2*np.pi*a*rbar/l)
    u = gamma*u/rbar
    return u


def hansen_cds(gamma, a, l, rbar):
    """The hansen qsp [1]_ taking into account quantum diffraction and fermionic
    symmetry.

    Parameters
    ----------
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        The thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The potential at the given sample points.

    References
    ----------
    .. [1] Hansen, J. P., and I. R. McDonald. “Microscopic Simulation of a
       Strongly Coupled Hydrogen Plasma.” Physical Review A 23, no. 4
       (April 1, 1981): 2041–59. https://doi.org/10.1103/PhysRevA.23.2041.
    """
    x = 2*np.pi*a*rbar/l
    u = 1 - np.exp(-1*x)
    u = gamma*u/rbar
    u = u + np.log(2)*np.exp(-1*x*x/np.log(2))
    return u


def kelbg_cd(gamma, a, l, rbar):
    """Another qsp from [1]_ taking into account quantum diffraction.

    Parameters
    ----------
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        The thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The potential at the given sample points.

    References
    ----------
    .. [1] Jones, Christopher S., and Michael S. Murillo. “Analysis of
           Semi-Classical Potentials for Molecular Dynamics and Monte Carlo
           Simulations of Warm Dense Matter.” High Energy Density Physics 3, no.
           3–4 (October 2007): 379–94. https://doi.org/10.1016/j.hedp.2007.02.038.
    """
    x = a*rbar*np.sqrt(2*np.pi)/l
    c = x*np.sqrt(np.pi)
    u = 1.0 - np.exp(-1*x*x) + c*erfc(x)
    u = gamma*u/rbar
    return u


def kelbg_cds(gamma, a, l, rbar):
    """Another qsp from [1]_ taking into account quantum diffraction and
    fermionic symmetry via an uhlman gropper term.

    Parameters
    ----------
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        The thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The potential at the given sample points.

    References
    ----------
    .. [1] Jones, Christopher S., and Michael S. Murillo. “Analysis of
           Semi-Classical Potentials for Molecular Dynamics and Monte Carlo
           Simulations of Warm Dense Matter.” High Energy Density Physics 3, no.
           3–4 (October 2007): 379–94. https://doi.org/10.1016/j.hedp.2007.02.038.
    """
    x = a*rbar*np.sqrt(2*np.pi)/l
    c = x*np.sqrt(np.pi)
    ug_term = 1 - 0.5*np.exp(-1*x*x)
    ug_term = np.log(ug_term)
    u = 1.0 - np.exp(-1*x*x) + c*erfc(x)
    u = gamma*u/rbar
    u = u - ug_term
    return u


def get_potential(key, gamma, a, l, rbar):
    """A wrapper function for getting any of the above potentials by providing
    a key.

    Parameters
    ----------
    key : string
        The key uniquely specifying which potential to get.
    gamma : float
        The coupling parameter for the pair interaction.
    a : float
        Ion radius.
    l : float
        Thermal wavelength.
    rbar : array of float
        The values (in units of a) at which to sample the potential.

    Returns
    -------
    u : array of float
        The temperature normalized potential at the given sample points."""
    u = np.zeros(rbar.shape[0])
    if key == 'c':
        u = coul(gamma, a, l, rbar)
    elif key == 'hcd':
        u = hansen_cd(gamma, a, l, rbar)
    elif key == 'hcds':
        u = hansen_cds(gamma, a, l, rbar)
    elif key == 'kcd':
        u = kelbg_cd(gamma, a, l, rbar)
    elif key == 'kcds':
        u = kelbg_cds(gamma, a, l, rbar)
    return u
