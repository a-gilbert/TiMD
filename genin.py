'''A script used to generate initial conditions and input script for an md
simulation consisting of Aluminum, Gold, and electrons. All inputs are assumed
to be in metal units.'''
import sys
import math
import random


class PositionGenerator:
    """A class that generates the positions for all particles in the
    simulation.

    This class generates positions for aluminum and gold ions subject to the
    constraint that ions are separated by a radius of at least ndens^{1/3}. The
    two species are also partitioned into the left and right sides of the sim
    domain. Additionally it generates the integer number of electrons for each
    species based off of the input temp.


    Attributes
    ----------
    xs : list of float
        The position of each particle in the x direction.
    ys : list of float
        The position of each particle in the y direction.
    zs : list of float
        The position of each particle in the z direction.
    atypes : list of int
        The type of each particle.
    aids : list of int
        The id number of each particle.
    charges : list of float
        The charge of each particles, in units of fundamental charge.
    temps : list of float
        The temperature of each particle, in units of kelvin.
    sim_domain : list of list of float
        The bottom rear left and top forward right coordinates of
        the sim domain.
    al_domain : list of list of float
        The bottom rear left and top forward right coordinates of
        the aluminum domain.
    au_domain : list of list of float
        The bottom rear left and top forward right coordinates of
        the gold domain.
    zbar : dict of int
        The integer roof of the ionization state for each ion species.
    nparticles : dict of int
        The total number particles of each type in the simulation.
    """

    def __init__(self, dx, dy, dz, fn='data.timd', Ti=10.0, Te=100.0):
        """A function to initialize a position generator instance.

        This function generates the lists of positions, types, and ids
        for each particle in a simulation.

        Parameters
        ----------
        dx : float
            The extent of the simulation in the x direction.
        dy : float
            The extent of the simulation in the y direction.
        dz : float
            The extent of the simulation in the z direction.
        fn : string, optional
            The file to save the generated data to.
        Ti : float, optional
            The temperature of the ions, in eV, to determine electron numbers.
            Defaults to 10 eV.
        Te : float, optional
            The temperature of the electrons in eV.
        """
        self.sim_domain = [[0, 0, 0], [dx, dy, dz]]
        self.al_domain = [[0, 0, 0], [dx, dy, 0.5*dz]]
        self.au_domain = [[0, 0, 0.5*dz],
                          [dx, dy, dz]]
        self.xs = []
        self.ys = []
        self.zs = []
        self.atypes = []
        self.ids = []
        self.temps = []
        self.charges = []
        self.zbar = {'Al': 0.0, 'Au': 0.0}
        self.nparticles = {'Al': 0, 'Au': 0, 'e': 0}

        self.set_zbar(Ti)
        self.set_nparticles()
        self.set_positions()
        self.set_charges()
        self.set_temps(Ti, Te)
        self.write_data(fn)

    def set_charges(self):
        for i in range(len(self.atypes)):
            if self.atypes[i] == 1:
                self.charges.append(-1.0)
            elif self.atypes[i] == 2:
                self.charges.append(float(self.zbar['Al']))
            elif self.atypes[i] == 3:
                self.charges.append(float(self.zbar['Au']))

    def set_temps(self, Ti, Te):
        inv_kb = 1.0/(8.6173303e-5)
        for i in range(len(self.atypes)):
            if self.atypes[i] == 1:
                self.temps.append(inv_kb*Te)
            elif self.atypes[i] == 2 or self.atypes[i] == 3:
                self.temps.append(inv_kb*Ti)

    def set_zbar(self, Ti):
        """Calculates the integer number of electrons unbound from each ion
        at standard density and the input temperature.

        Parameters
        ----------
        Ti : float
            The ion temperature, in eV, for the simulation

        Notes
        -----
        Code replicated from https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database/blob/master/database/Thomas_Fermi_Ionization/Thomas_Fermi_Z_Python.ipynb
        """
        alpha = 14.3139
        beta = 0.6624
        a1 = 0.003323
        a2 = 0.9718
        a3 = 9.26148e-5
        a4 = 3.1065
        b0 = -1.7630
        b1 = 1.43175
        b2 = 0.31546
        c1 = -0.366667
        c2 = 0.983333

        # room temperature densities for aluminum and gold in g/cm^{3}.
        rho = {'Al': 2.70, 'Au': 19.30}
        z = {'Al': 13, 'Au': 79}
        atomic_mass = {'Al': 26.9815385, 'Au': 196.966569}

        for i in ['Al', 'Au']:
            R = rho[i]/(z[i]*atomic_mass[i])
            T0 = Ti/z[i]**(4./3.)
            Tf = T0/(1+T0)
            A = a1*T0**a2 + a3*T0**a4
            B = -1*math.exp(b0 + b1*Tf + b2*Tf**7)
            C = c1*Tf + c2
            Q1 = A*R**B
            Q = (R**C+Q1**C)**(1/C)
            x = alpha*Q**beta
            zbar = z[i]*x/(1 + x + math.sqrt(1 + 2.*x))
            self.zbar[i] = int(math.ceil(zbar))

    def set_nparticles(self):
        """Sets the total number of particles of each type in the simulation.
        """
        n_avagadro = 6.022140857e23

        # g/Mol
        molMass_Al = 26.9815385
        molMass_Au = 196.966569

        mass_Al = molMass_Al/n_avagadro
        mass_Au = molMass_Au/n_avagadro
        # g/cm^3
        rho_Al = 2.70
        rho_Au = 19.30
        # conversion constant from cubic centimeters to cubic angstrom
        c_cm3a3 = 1e-24
        # number per cubic angstrom
        ndens_Al = (rho_Al/mass_Al)*c_cm3a3
        ndens_Au = (rho_Au/mass_Au)*c_cm3a3
        v_al = 1.0
        v_au = 1.0
        for i in range(3):
            v_al = v_al*(self.al_domain[1][i] - self.al_domain[0][i])
            v_au = v_au*(self.au_domain[1][i] - self.au_domain[0][i])
        self.nparticles['Al'] = int(math.floor(ndens_Al*v_al))
        self.nparticles['Au'] = int(math.floor(ndens_Au*v_au))
        self.nparticles['e'] = self.zbar['Au']*self.nparticles['Au'] \
            + self.zbar['Al']*self.nparticles['Al']

    def get_nunique_rands(self, n):
        """A convenience method to return n unique random floats between 0 and
        1"""
        out = []
        for i in range(2*n):
            out.append(random.random())
        out = random.sample(out, n)
        return out

    def get_dr(self, r0, rmax, l):
        """A convenience function to determine the maximum allowed change in
        a unit cell."""
        dr = [0, 0, 0]
        for i in range(3):
            c = r0[i] + l
            if c < rmax[i]:
                dr[i] = l
            else:
                dr[i] = rmax[i] - r0[i]
        return dr

    def deposit_electrons(self, n, z, r0, dr, rr):
        """A method to deposit z electrons all within a cube of length l from
        x0, y0, z0.

        Parameters
        ----------
        n : int
            The first electron id.
        z : int
            Number of electrons being added.
        r0 : list of float
            The bottom rear left corner of the cell.
        dr : list of float
            The allowed change in length in each direction for a cell.
        rr : list of list of float
            The list of random scaling factors in each direction applied to dr
            for each particle.
       """
        for i in range(z):
            self.ids.append(n + i)
            self.atypes.append(1)
            self.xs.append(r0[0] + dr[0]*rr[0][i])
            self.ys.append(r0[0] + dr[1]*rr[1][i])
            self.zs.append(r0[0] + dr[2]*rr[2][i])

    def deposit_aluminum_ion(self, n, r0, dr, rr):
        """A method to deposit a single aluminum ion into storage.

        Parameters
        ----------
        n : int
            The particle id.
        r0 : list of float
            The left bottom rear point of the cell.
        dr : list of float
            The maximum amount of allowed change in each direction.
        rr : list of float
            The scaling factor applied to dr in each dimension.
       """
        self.ids.append(n)
        self.atypes.append(2)
        self.xs.append(r0[0] + dr[0]*rr[0])
        self.ys.append(r0[1] + dr[1]*rr[1])
        self.zs.append(r0[2] + dr[2]*rr[2])

    def deposit_gold_ion(self, n, r0, dr, rr):
        """A method to deposit a single gold ion into storage.

        Parameters
        ----------
        n : int
            The particle id.
        r0 : list of float
            The left bottom rear point of the cell.
        dr : list of float
            The maximum amount of allowed change in each direction.
        rr : list of float
            The scaling factor applied to dr in each dimension.
       """
        self.ids.append(n)
        self.atypes.append(3)
        self.xs.append(r0[0] + dr[0]*rr[0])
        self.ys.append(r0[1] + dr[1]*rr[1])
        self.zs.append(r0[2] + dr[2]*rr[2])

    def deposit_aluminum(self, n, r0, dr):
        """Deposits an aluminum ion and its electrons at unique positions in a
        rectangle with bottom rear left corner centered at r0 and top forward
        right edge at r0 + dr.

        Parameters
        ----------
        n : int
            The nucleus id.
        r0 : list of float
            The rear left bottom point of the cell in question.
        dr : list of float
            The distance to the top right forward point of the cell.

        Returns
        -------
        n : int
            The id value of the next particle to be deposited.
        """
        n_tot = 1 + self.zbar['Al']
        xr = self.get_nunique_rands(n_tot)
        yr = self.get_nunique_rands(n_tot)
        zr = self.get_nunique_rands(n_tot)
        self.deposit_aluminum_ion(n, r0, dr, [xr[0], yr[0], zr[0]])
        self.deposit_electrons(n+1, self.zbar['Al'], r0, dr,
                               [xr[1:], yr[1:], zr[1:]])
        return n + n_tot

    def deposit_gold(self, n, r0, dr):
        """Deposits a gold ion and its electrons at unique positions in a
        rectangle with bottom rear left corner centered at r0 and top forward
        right edge at r0 + dr.

        Parameters
        ----------
        n : int
            The nucleus id.
        r0 : list of float
            The rear left bottom point of the cell in question.
        dr : list of float
            The distance to the top right forward point of the cell.

        Returns
        -------
        n : int
            The id value of the next particle to be deposited.
        """
        n_tot = 1 + self.zbar['Au']
        xr = self.get_nunique_rands(n_tot)
        yr = self.get_nunique_rands(n_tot)
        zr = self.get_nunique_rands(n_tot)
        self.deposit_gold_ion(n, r0, dr, [xr[0], yr[0], zr[0]])
        self.deposit_electrons(n+1, self.zbar['Au'], r0, dr,
                               [xr[1:], yr[1:], zr[1:]])
        return n + n_tot

    def set_aluminum_positions(self):
        """ A function to generate the positions of the aluminum ions and
        the corresponding electrons."""
        x0 = self.al_domain[0][0]
        y0 = self.al_domain[0][1]
        z0 = self.al_domain[0][2]

        xmax = self.al_domain[1][0]
        ymax = self.al_domain[1][1]
        zmax = self.al_domain[1][2]

        rmax = [xmax, ymax, zmax]

        l_al = (xmax-x0)*(ymax-y0)*(zmax-z0)
        l_al = l_al/self.nparticles['Al']
        l_al = math.pow(l_al, 1./3.)

        n = 1
        n_tot = self.nparticles['Al']*(1 + self.zbar['Al'])
        while x0 < xmax:
            while y0 < ymax:
                while z0 < zmax:
                    if n < n_tot:
                        r0 = [x0, y0, z0]
                        dr = self.get_dr(r0, rmax, l_al)
                        n = self.deposit_aluminum(n, r0, dr)
                    z0 += l_al
                z0 = self.al_domain[0][2]
                y0 += l_al
            y0 = self.al_domain[0][1]
            z0 = self.al_domain[0][2]
            x0 += l_al

    def set_gold_positions(self):
        """ A function to generate the positions of the gold ions and
        the corresponding electrons."""
        x0 = self.au_domain[0][0]
        y0 = self.au_domain[0][1]
        z0 = self.au_domain[0][2]

        xmax = self.au_domain[1][0]
        ymax = self.au_domain[1][1]
        zmax = self.au_domain[1][2]
        rmax = [xmax, ymax, zmax]

        l_au = (xmax-x0)*(ymax-y0)*(zmax-z0)
        l_au = l_au/self.nparticles['Au']
        l_au = math.pow(l_au, 1./3.)

        n = self.ids[-1] + 1
        n_tot = self.nparticles['Au']*(1 + self.zbar['Au'])
        n_tot = n_tot + self.nparticles['Al']*(1 + self.zbar['Al'])
        while x0 < xmax:
            while y0 < ymax:
                while z0 < zmax:
                    if n < n_tot:
                        r0 = [x0, y0, z0]
                        dr = self.get_dr(r0, rmax, l_au)
                        n = self.deposit_gold(n, r0, dr)
                    z0 += l_au
                z0 = self.au_domain[0][2]
                y0 += l_au
            x0 += l_au
            y0 = self.au_domain[0][1]
            z0 = self.au_domain[0][2]

    def set_positions(self):
        self.set_aluminum_positions()
        self.set_gold_positions()

    def write_data(self, fn):
        """Writes all of the generated data to file."""
        f = open(fn, 'w')
        f.write("#TiMD Input Data File\n")
        f.write("\n")
        n_tot = self.nparticles['Al']*(1+self.zbar['Al'])
        n_tot = n_tot + self.nparticles['Au']*(1 + self.zbar['Au'])
        f.write("%d atoms\n" % n_tot)
        f.write("3 atom types\n")
        bounds = [['xlo', 'ylo', 'zlo'],
                  ['xhi', 'yhi', 'zhi']]
        for i in range(3):
            s = "%.16e %.16e %s %s\n" % (self.sim_domain[0][i],
                                         self.sim_domain[1][i],
                                         bounds[0][i], bounds[1][i])
            f.write(s)
        f.write("\n")
        f.write("Atoms\n")
        f.write("\n")
        for i in range(len(self.ids)):
            s = "%d %d %.16e %.16e %.16e %.16e %.16e\n" % (self.ids[i],
                                                           self.atypes[i],
                                                           self.charges[i],
                                                           self.temps[i],
                                                           self.xs[i],
                                                           self.ys[i],
                                                           self.zs[i])
            f.write(s)
        f.close()


def main():
    if len(sys.argv) == 4:
        PositionGenerator(float(sys.argv[1]), float(sys.argv[2]),
                          float(sys.argv[3]))
    quit()


if __name__ == "__main__":
    main()
