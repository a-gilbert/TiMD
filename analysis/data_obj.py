"""A set of classes to facilitate loading and manipulating
LAMMPS molecular dynamics data."""

import numpy as np


class SimData:
    """A class to load and manipulate LAMMPS MD data.

    This class and its methods are designed with the TiMD simulation in
    particular.

    Parameters
    ----------
    fn : string
        The name of a lammps output file to load.
    periodic : Optional, list of bool
        Whether or not the simulation is periodic in each cartesian direction.

    Attributes
    ----------
    xs : array of float
        The array of particle x values.
    ys : array of float
        The array of particle y values.
    zs : array of float
        The array of particle z values.
    vxs : array of float
        Array of particle x velocities.
    vys : array of float
        Array of particle y velocities.
    vzs : array of float
        Array of particle z velocities.
    fxs : array of float
        Array of particle x forces.
    fxs : array of float
        Array of particle y forces.
    fxs : array of float
        Array of particle z forces.
    ids : array of int
        Array of particle ids.
    temps : array of float
        Array of particle temperatures.
    ptypes : list of int
        A list giving the type of each particle.
    ntypes : dict
        A dict giving the number of each type of particle.
    masses : dict
        A dict of masses for each type.
    charge : dict
        A dict of charge values for each type.
    bbox : list of array of float
        The extent of the simulation domain
    period : list of bool
        Whether or not the simulation is periodic in a cartesian direction.
    units : dict
        What units are used for any physical quantity.
    timestep : int
        What timestep was this output generated on
    """

    def __init__(self, fn, periodic=[True, True, True]):
        self.xs = []
        self.ys = []
        self.zs = []
        self.vxs = []
        self.vys = []
        self.vzs = []
        self.fxs = []
        self.fys = []
        self.fzs = []
        self.ids = []
        self.temps = []
        self.ptypes = []
        self.type_names = {'e': 1, 'Al': 2, 'Au': 3}
        self.masses = dict(e=0.0, Al=0.0, Au=0.0)
        self.charge = dict(e=-1.0, Al=3.0, Au=7.0)
        self.ntypes = dict(e=0, Al=0, Au=0, All=0)
        self.bbox = [[0, 0, 0], [0, 0, 0]]
        self.period = periodic
        self.timestep = 0
        self.units = {'mass': 'g/mol',
                      'length': 'A',
                      'time': 'ps',
                      'energy': 'eV',
                      'temperature': 'K',
                      'force': 'eV/A',
                      'velocity': 'A/ps',
                      'charge': 'e'}

        mass_set = dict(e=False, Al=False, Au=False)
        charge_set = dict(e=False, Al=False, Au=False)

        f_in = open(fn, 'r')
        f_in = f_in.readlines()
        self.timestep = int(f_in[1])
        self.ntypes['All'] = int(f_in[3])
        for i in range(3):
            line = f_in[5 + i].split()
            self.bbox[0][i] = float(line[0])
            self.bbox[1][i] = float(line[1])
        for i in range(0, len(f_in) - 9):
            line = f_in[i + 9].split()
            self.ids.append(int(line[0]))
            self.ptypes.append(int(line[1]))
            if not mass_set['e'] and self.ptypes[i] == 1:
                self.masses['e'] = float(line[2])
                mass_set['e'] = True
            if not charge_set['e'] and self.ptypes[i] == 1:
                self.charge['e'] = float(line[3])
                charge_set['e'] = True
            if not mass_set['Al'] and self.ptypes[i] == 2:
                self.masses['Al'] = float(line[2])
                mass_set['Al'] = True
            if not charge_set['Al'] and self.ptypes[i] == 2:
                self.charge['Al'] = float(line[3])
                charge_set['Al'] = True
            if not mass_set['Au'] and self.ptypes[i] == 3:
                self.masses['Au'] = float(line[2])
                mass_set['e'] = True
            if not charge_set['Au'] and self.ptypes[i] == 3:
                self.charge['Au'] = float(line[3])
                charge_set['Au'] = True
            self.temps.append(float(line[4]))
            self.xs.append(float(line[5]))
            self.ys.append(float(line[6]))
            self.zs.append(float(line[7]))
            self.vxs.append(float(line[8]))
            self.vys.append(float(line[9]))
            self.vzs.append(float(line[10]))
            self.fxs.append(float(line[11]))
            self.fys.append(float(line[12]))
            self.fzs.append(float(line[13]))
        self.temps = np.array(self.temps)
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.zs = np.array(self.zs)
        self.vxs = np.array(self.vxs)
        self.vys = np.array(self.vys)
        self.vzs = np.array(self.vzs)
        self.fxs = np.array(self.fxs)
        self.fys = np.array(self.fys)
        self.fzs = np.array(self.fzs)
        self.ids = np.array(self.ids)
        self.ptypes = np.array(self.ptypes)

    def get_lneighbors(self, neigh, x0, y0, z0, r, t):
        """Get all particles of type t within a radius r of point r0 that
        are not image particles."""
        rs = (self.xs - x0)**2 + (self.ys - y0)**2 + (self.zs - z0)**2
        rs = np.sqrt(rs)
        rs = rs[self.ptypes == self.type_names[t]]
        lids = self.ids[self.ptypes == self.type_names[t]]
        neigh['ids'].append(lids[rs < r].to_list())

    def get_img_lims(self, x0, y0, z0, r):
        """Determine how far in each direction image domains must extend for
        the given r."""
        dr_box = []
        for i in range(3):
            dr_box.append(self.bbox[1][i] - self.bbox[0][i])
        drp2h = [dr_box[0] - x0, dr_box[1] - y0, dr_box[2] - z0]
        drp2l = [x0 - self.bbox[0][0], y0 - self.bbox[0][1], z0 -
                 self.bbox[0][2]]
        nlims = [[0, 0, 0], [0, 0, 0]]
        # x hi and lo limits
        while drp2l[0] - dr_box[0]*nlims[0][0] < r:
            nlims[0][0] -= 1
        while drp2h[0] + nlims[1][0]*dr_box[0] < r:
            nlims[1][0] += 1
        # y hi and lo limits
        while drp2l[1] - dr_box[1]*nlims[0][1] < r:
            nlims[0][1] -= 1
        while drp2h[1] + nlims[1][1]*dr_box[1] < r:
            nlims[1][1] += 1
        # z hi and lo limits.
        while drp2l[2] - dr_box[2]*nlims[0][2] < r:
            nlims[0][2] -= 1
        while drp2h[2] + dr_box[2]*nlims[1][2] < r:
            nlims[1][2] += 1
        return nlims, drp2l, dr_box

    def get_img_neighbors(self, neigh, x0, y0, z0, n, drp2l, dr_box, r, t):
        r1 = []
        for i in range(3):
            r1.append(n[i]*dr_box[i] - drp2l[i])
        rs = (self.xs - self.bbox[0][0] + r1[0])**2
        rs = rs + (self.ys - self.bbox[0][1] + r1[1])**2
        rs = rs + (self.zs - self.bbox[0][2] + r1[2])**2
        rs = np.sqrt(rs)
        rs = rs[self.ptypes == self.type_names[t]]
        iids = self.ids[self.ptypes == self.type_names[t]]
        neigh['ids'].append(iids[rs < r].to_list())

    def get_pneighbors(self, x0, y0, z0, r, t):
        """Get all of the particles of type t within a radius r of a point x0,
        y0, z0, accounting for the fact that the sim domain is periodic."""
        neigh = {'num': 0, 'ids': []}
        # determine how far in each direction the image domains must extend for
        # the given r
        nlims, drp2l, dr_box = self.get_img_lims(x0, y0, z0, r)
        for i in range(nlims[0][0], nlims[1][0] + 1):
            for j in range(nlims[0][1], nlims[1][1] + 1):
                for k in range(nlims[0][2], nlims[1][2] + 1):
                    if i == 0 and j == 0 and k == 0:
                        self.get_lneighbors(neigh, x0, y0, z0, r, t)
                    else:
                        self.get_img_neighbors(neigh, x0, y0, z0, [i, j, k],
                                               drp2l, dr_box, r, t)
        neigh['num'] = len(neigh['ids'])
        return neigh

    def get_avg_ndens(self, t, r):
        mask = self.ptypes == self.type_names[t]
        xs = self.xs[mask]
        ys = self.ys[mask]
        zs = self.zs[mask]
        ids = self.ids[mask]
        avg_ndens = 0
        v = 4*np.pi*r*r*r/3.0
        for i in range(len(ids)):
            neigh = self.get_pneighbors(xs[i], ys[i], zs[i], r, t)
            avg_ndens += neigh['num']/v
        avg_ndens = avg_ndens/len(ids)
        return avg_ndens

    def get_temp(self, x0, y0, z0, r, t):
        """Returns temperature, IN UNITS OF eV!"""
        neigh = self.get_pneighbors(x0, y0, z0, r, t)
        temp = 0
        for i in neigh['ids']:
            mask = self.ids == i
            temp += self.vxs[mask]*self.vxs[mask]
            temp += self.vys[mask]*self.vys[mask]
            temp += self.vzs[mask]*self.vzs[mask]
        temp = temp*self.masses[t]*3.4547563333333336e-5
        temp = temp/neigh['num']
        return temp

    def get_avg_temp(self, r, t):
        mask = self.ptypes == self.type_names[t]
        xs = self.xs[mask]
        ys = self.ys[mask]
        zs = self.zs[mask]
        ids = self.ids[mask]
        num_part = len(ids)
        avg_temp = 0
        for i in range(num_part):
            avg_temp += self.get_temp(xs[i], ys[i], zs[i], r, t)
        return avg_temp





    def get_avg_temp(self, t, r):
        xs = self.xs[self.ptypes == t]
        ys = self.ys[self.pytpes == t]
        zs = self.zs[self.ptypes == t]
        ids = self.ids[self.ptypes == t]
        avg_temp = 
