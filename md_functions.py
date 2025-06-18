# Things to do:
# - include Fourier-based propagation
# - move force_pa to parent class (low priority)

import glob
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

from numba import njit
import numpy as np
from numpy.typing import NDArray

# Type alias for clearer annotations
FloatArray = NDArray[np.float64]

# # # # # # # # # # # # # # # # # # #
#     M D   P A R A M E T E R S     #
# # # # # # # # # # # # # # # # # # #

@dataclass
class MDParams:
    n_pre_run: int = 0
    n_run: int = 1
    n_relax: int = 400
    n_observe: int = 600
    n_obs_max: int = 10_000_000
    temp: float = 1.0
    dt: float = 0.01
    box_x: float = 0.0
    i_run: int = field(default=0, init=False, repr=False)  # integer run
    i_time: int = field(default=0, init=False, repr=False) # integer time
    r_time: float = field(default=0., repr=False)   # real time


# # # # # # # # # # # # # # # # #
#     P R O P A G A T O R S     #
# # # # # # # # # # # # # # # # #

@dataclass
class Propagator(ABC):

    n_sample: int
    n_atom: int
    mdp: MDParams = field(repr=False)

    f_const_vel: bool = False
    const_vel: float = np.nan
    const_idx: Union[int, np.ndarray, None] = None

    name: str = ""

    # Initialize with empty arrays instead of None
    x: FloatArray = field(default_factory=lambda: np.array([]), repr=False)
    f: FloatArray = field(default_factory=lambda: np.array([]), repr=False)

    def __post_init__(self):
        # make sure that child has assigned a name
        if not self.name:
            raise ValueError("Propagator child must assign name")

        # Check if arrays are empty (size = 0) rather than None
        if self.x.size == 0:
            self.x = self.mdp.box_x * (np.arange(self.n_atom)+0.5)/self.n_atom \
                   * np.ones((self.n_sample, self.n_atom))
            
        if self.f.size == 0:
           self.f = np.zeros_like(self.x)

    @abstractmethod
    def init_coeffs(self):
        pass

    @abstractmethod
    def propagate(self):
        pass

    def pre_propagate(self):
        if not self.f_const_vel:
            return

        if self.mdp.i_run == -self.mdp.n_pre_run and self.mdp.i_time == -self.mdp.n_relax:
            if isinstance(self.const_idx,np.ndarray):
                self.x_old = np.zeros([self.n_sample, len(self.const_idx)])
            else:
                self.x_old = np.zeros(self.n_sample)

        if self.const_idx is None:
            self.x_old[:] = np.average(self.x, axis=-1)
        else:
            self.x_old[:] = self.x[:,self.const_idx]

    def post_propagate(self):
        if not self.f_const_vel:
            return

        dx_target = self.const_vel * self.mdp.dt
        if self.const_idx is None:
            self.x += (dx_target - np.average(self.x, axis=-1, keepdims=True) 
                       + self.x_old[:,None])
        else:
            self.x[:, self.const_idx] = dx_target + self.x_old


@dataclass
class StochProp(Propagator):
    """Thermostats needing one r.v. for each atom in each sample."""

    f_rand: str = "gauss"
    g: FloatArray = field(default_factory=lambda: np.array([]), repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.g.size == 0:
            self.g = np.zeros_like(self.x)
            set_random(self.g, self.f_rand)


@dataclass
class Brown(StochProp):
    dampg: float = 1.0
    name: str = field(default="br", init=False)

    def init_coeffs(self): 
        self.c_xf = self.mdp.dt / self.dampg
        self.c_xg = np.sqrt(self.mdp.temp * self.c_xf / 2)
    
    def propagate(self):
        self.pre_propagate()
        self.x += self.c_xg * self.g
        set_random(self.g, self.f_rand)
        self.x += self.c_xg * self.g
        self.x += self.c_xf * self.f 
        self.post_propagate()


@dataclass
class Langevin(StochProp):
    mass: float = 1.0
    tau: float = 0.5
    name: str = field(default="lol", init=False)
    
    v: FloatArray = field(default_factory=lambda: np.array([]), repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.v.size == 0:
            v_th = np.sqrt(self.mdp.temp/self.mass)
            self.v = v_th * np.random.randn(*self.x.shape)
            self.t_kin = np.zeros_like(self.x)

    def init_coeffs(self):
        dt, temp = self.mdp.dt, self.mdp.temp
        self.c_vv = 1. - dt / self.tau
        self.c_vf = dt / self.mass
        self.c_vg = np.sqrt(2 * temp / self.mass * dt / self.tau)
        self.c_xv = dt
        self.check_coeffs()

    def check_coeffs(self):
        """
        Function resets to Verlet if dt/tau is too small and terminates if too large
        """
        dt_rel = self.mdp.dt / self.tau
        if dt_rel < 1.e-6:
            self.c_vv = 1.0
            self.c_xv = self.mdp.dt
            self.c_vf = self.c_xv / self.mass
            self.c_vg = 0.0
        elif dt_rel > 20.0:
            raise ValueError("Think of using Brownian thermostat instead")

    def propagate(self):
        self.pre_propagate()
        self.v *= self.c_vv
        self.v += self.c_vf * self.f + self.c_vg * self.g
        self.x += self.c_xv * self.v
        set_random(self.g, self.f_rand)
        self.post_propagate()


@dataclass
class BPCLangevin(Langevin):
    name: str = field(default="bpcl", init=False)
    
    def init_coeffs(self):
        dt, temp = self.mdp.dt, self.mdp.temp
        self.c_vv = np.exp(-dt / self.tau)
        self.c_xv = np.sqrt(2 * self.tau * dt * (1-self.c_vv) / (1+self.c_vv))
        self.c_vf = np.sqrt((1-self.c_vv**2) * self.tau * dt / 2) / self.mass
        self.c_vg = np.sqrt(temp / self.mass * (1 - self.c_vv**2))
        self.check_coeffs()

@dataclass
class GJLangevin(Langevin):
    name: str = field(default="gjl", init=False)
        
    def init_coeffs(self):
        dt, temp = self.mdp.dt, self.mdp.temp
        self.c_vv = np.exp(-dt / self.tau)
        self.c_xv = np.sqrt((1-self.c_vv) * self.tau * dt)
        self.c_vf = self.c_xv / self.mass
        self.c_vg = np.sqrt(temp / self.mass * (1 - self.c_vv) / 2)
        self.check_coeffs()

    def propagate(self):
        self.pre_propagate()
        self.v *= self.c_vv
        self.v += self.c_vf * self.f + self.c_vg * self.g
        set_random(self.g, self.f_rand)
        self.v += self.c_vg * self.g
        self.x += self.c_xv * self.v
        self.post_propagate()

@dataclass
class MCLangevin(BPCLangevin):
    name: str = field(default="mcl", init=False)
    n_atom_per_bin:int = 2
    n_bin:int = field(init=False, repr=False)

    def __post_init__(self):
        self.n_bin = self.n_atom // self.n_atom_per_bin
        if self.n_bin * self.n_atom_per_bin != self.n_atom:
            raise ValueError("n_atom_per_bin must be a divisor of n_atom")
        super().__post_init__()
        bin_shape = (self.n_sample, self.n_bin)
        self.x_com = np.zeros(bin_shape)
        self.v_com = np.zeros(bin_shape)
        self.f_com = np.zeros(bin_shape)

    def init_coeffs(self):
        super().init_coeffs()
        self.c_vf_com = self.mdp.dt / self.mass
        self.c_xv_com = self.mdp.dt

    def propagate(self):
        self.pre_propagate()
        n_apb = self.n_atom_per_bin

        # pick random first atom to emulate randomly placed binning boxes
        off_set = np.random.randint(0, n_apb)

        for i_bin in range(self.n_bin):

            indices = np.arange(i_bin*n_apb+off_set, (i_bin+1)*n_apb+off_set) 
            indices[-off_set:] = indices[-off_set:] - self.n_atom

            # compute com and relative x, v, and f
            self.x_com[:, i_bin] = np.mean(self.x[:, indices], axis=1)
            self.x[:, indices] -= self.x_com[:, i_bin][:, np.newaxis]
            self.v_com[:, i_bin] = np.mean(self.v[:, indices], axis=1)
            self.v[:, indices] -= self.v_com[:, i_bin][:, np.newaxis]
            self.f_com[:, i_bin] = np.mean(self.f[:, indices], axis=1)
            self.f[:, indices] -= self.f_com[:, i_bin][:, np.newaxis]

            # propagate relative positions similarly as in BPCL
            self.v[:, indices] *= self.c_vv
            self.v[:, indices] += self.c_vf * self.f[:, indices] 
            self.v[:, indices] += self.c_vg * (self.g[:, indices] 
                                - np.mean(self.g[:, indices], axis=1, keepdims=True))
            self.x[:, indices] += self.c_xv * self.v[:, indices]

        # propagate com according to Verlet
        self.v_com += self.c_vf_com * self.f_com
        self.x_com += self.c_xv_com * self.v_com

        # add com back on to relative positions and velocities
        for i_bin in range(self.n_bin):

            indices = np.arange(i_bin*n_apb+off_set, (i_bin+1)*n_apb+off_set) 
            indices[-off_set:] = indices[-off_set:] - self.n_atom

            self.x[:, indices] += self.x_com[:, i_bin][:, np.newaxis]
            self.v[:, indices] += self.v_com[:, i_bin][:, np.newaxis]

        set_random(self.g, self.f_rand)

        self.post_propagate()
        
# Function returns a Maxwell object, whose class inherits dynamically from propagator
def maxwell(propagator):
    
    @dataclass
    class Maxwell(propagator):
    
        n_mxw: int = 1
        name: str = field(default="mxw", init = False)
        

        def __init__(self, k_mxw, tau_mxw, *args, **kwargs):
            super().__init__(*args, **kwargs) 
            self.k_mxw = k_mxw
            self.tau_mxw = tau_mxw
            
            if len(k_mxw) != len(tau_mxw):
                raise ValueError("The lengths of k_mxw and tau_mxw arrays don't match.")
            self.n_mxw = len(k_mxw)
            
            self.name = f"mxw_{propagator.name}"
            self.x_mxw = np.zeros((self.n_sample,self.n_atom,self.n_mxw), dtype=float)
            self.g_mxw = np.zeros((self.n_sample,self.n_atom,self.n_mxw), dtype=float)
            self.f_mxw = np.zeros((self.n_sample,self.n_atom,self.n_mxw), dtype=float)
            self.f_mxw[:] = np.nan
            dx_thermal = np.sqrt(self.mdp.temp/self.k_mxw)
            for ii in range(self.n_mxw):
                self.x_mxw[:,:,ii] = self.x + \
                    dx_thermal[ii] * np.random.randn(self.n_sample,self.n_atom)
                set_random(self.g_mxw[:,:,ii], self.f_rand)

        def init_coeffs(self):
            dt, temp = self.mdp.dt, self.mdp.temp
            self.temp = temp
            self.dampg_mxw = self.k_mxw * self.tau_mxw
            self.c_mxw_xf = np.divide(dt, self.dampg_mxw)
            self.c_mxw_xg = np.sqrt(temp * self.c_mxw_xf /2)
            super().init_coeffs()
            self.c_vv = 1.0
            self.c_xv = dt
            self.c_vg = 0.0
        
        def propagate(self):

            # force calculation needs to be done first
            for ii in range(self.n_mxw):
                self.f_mxw[:,:,ii] = self.k_mxw[ii] * (self.x - self.x_mxw[:,:,ii])

            self.f -= np.sum(self.f_mxw, axis=-1)

            for ii in range(self.n_mxw):
                self.x_mxw[:,:,ii] += self.c_mxw_xg[ii] * self.g_mxw[:,:,ii]

            set_random(self.g_mxw, self.f_rand)

            for ii in range(self.n_mxw):
                self.x_mxw[:,:,ii] += self.c_mxw_xg[ii] * self.g_mxw[:,:,ii]
                self.x_mxw[:,:,ii] += self.c_mxw_xf[ii] * self.f_mxw[:,:,ii]
                
            super().propagate()

    return Maxwell


# # # # # # # # # # # # # # # #
#     P O T E N T I A L S     #
# # # # # # # # # # # # # # # #

@dataclass
class Potential(ABC):

    n_sample: int
    n_atom: int
    box_x: float = 0.0
    f_single_part: bool = False
    name: str = field(default="", init=False)

    def __post_init__(self):

        if not self.f_single_part and not self.box_x:
            raise ValueError("set box_x for many-body potential")

        if self.f_single_part:
            self.v_pot = np.full((self.n_sample, self.n_atom), np.nan, dtype='float')
        else:
            self.v_pot = np.zeros(self.n_sample, dtype='float')

        self.v_pot_1 = np.zeros(self.n_sample, dtype='float')
        self.v_pot_2 = np.zeros(self.n_sample, dtype='float')

    @abstractmethod
    def set_force(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_v_pot(self, *args, **kwargs):
        pass

    def set_v_pot_mmt(self, config):
        self.set_v_pot(config)
        if self.f_single_part:
            self.v_pot_1[:] = np.mean(self.v_pot, axis=-1)
            self.v_pot_2[:] = np.mean(self.v_pot * self.v_pot, axis=-1)
        else:
            self.v_pot_1[:] = self.v_pot
            self.v_pot_2[:] = self.v_pot**2

@dataclass
class HarmOsc(Potential):
    k_spring: float = 1.0
    name: str = field(default="ho", init=False)
    f_single_part: bool = field(default=True, init=False)

    def set_force(self, config):
        x, f = config[0], config[1]
        f[:] = -self.k_spring * x

    def set_v_pot(self, config):
        x = config[0]
        self.v_pot[:] = self.k_spring * x * x / 2

@dataclass
class SinusSub(Potential):
    name: str = field(default="ss", init=False)
    f_single_part: bool = True
    
    def set_force(self, config):
        x, f = config[0], config[1]
        f[:] = np.sin(x)
    
    def set_v_pot(self, config):
        x = config[0]
        self.v_pot[:] = np.cos(x)

@dataclass      
class LennardJones(Potential):

    name: str = field(default="lj", init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.box_x == 0.0:
            raise ValueError("box_x must be finite")

    def set_force(self, config):
        x, f = config[0], config[1]
        if np.any(x[:, :-1] >= x[:, 1:]):
            raise ValueError(f"LJ atoms swapped place. pid: {os.getpid()}")
        add_lj_force(x, f, self.v_pot, self.box_x)

    def set_v_pot(self, config):
        config = config

@njit
def add_lj_force(x, force, v_pot, l_x):
    n_sample, n_atom = x.shape
    v_pot[:] = 0.0
    force[:] = 0.0

    for i_sample in range(n_sample):
        for i_atom in range(n_atom):
            j_atom = (i_atom + 1) % n_atom
            dx = x[i_sample, j_atom] - x[i_sample, i_atom]

            dx += (j_atom == 0) * l_x

            dx2 = dx * dx
            dx2 = max(dx2, 0.6) # leads to an energy of ≈12

            dx_inv_6 = 1.0 / (dx2 * dx2 * dx2)
            dx_inv_12 = dx_inv_6 * dx_inv_6

            v_pot[i_sample] += dx_inv_12 - 2.0 * dx_inv_6
            f_local = 12.0 * (dx_inv_12 - dx_inv_6) * dx / dx2

            force[i_sample, j_atom] += f_local
            force[i_sample, i_atom] -= f_local
    v_pot /= n_atom


@dataclass
class FKChain(Potential):

    n_well: int = 0
    k_spring: float = np.nan
    v0_sub: float = 1.0
    pbc: bool = True
    force_pa: float = np.nan
    force_idx: Union[int, np.ndarray, None] = None
    q: float = field(default=np.nan, init=False, repr=False)
    latt_const: float = field(default=np.nan, init=False, repr=False)
    name: str = field(default="fkc", init=False, repr=False)

    def __post_init__(self):
        self.latt_const = self.n_atom / self.n_well
        self.q = 2 * np.pi / self.latt_const
        super().__post_init__() 

    def set_force(self, config):
        x, f = config[0], config[1]

        # initialize force with substrate force
        f[:] = self.q * self.v0_sub * np.sin(self.q * x)
        
        # add external force if present
        if self.force_pa is not np.nan:
            if isinstance(self.force_idx, int):
                f[:,self.force_idx] += self.n_atom * self.force_pa
            elif isinstance(self.force_idx, np.ndarray):
                f[:,self.force_idx] += self.force_pa * self.n_atom/len(self.force_idx)
            else:
                f += self.force_pa

        # add elastic force
        add_elast_force(x, f, self.k_spring, self.pbc)

    def set_v_pot(self, config):
        x = config[0]
        self.v_pot[:] = elastic_energy(x,self.k_spring,self.pbc) \
                + self.v0_sub * np.sum(1 + np.cos(self.q*x), axis=-1)

@njit   
def add_elast_force(x, f, k_spring, f_pbc):
    df = k_spring * (x[:, 1:] - x[:, :-1] - 1.0)
    f[:, :-1] += df
    f[:, 1:] -= df 
    if not f_pbc:
        return
    n_atom = x.shape[1]
    df = k_spring * (x[:, 0] + n_atom - x[:, -1] - 1.0)
    f[:, -1] += df
    f[:, 0] -= df

@njit               
def elastic_energy(posit, k_spring, f_pbc):
    n_atom = posit.shape[1] 
    displ = (posit[:, 1:] - posit[:, :-1] - 1.0)
    v_pot = np.sum(displ**2, axis=1)
    if f_pbc:  # Only add periodic boundary contributions when f_pbc is True
        displ_pbc = posit[:, 0] + n_atom - posit[:, -1] - 1.0
        v_pot += displ_pbc**2
    return k_spring * v_pot / 2


# # # # # # # # # # # # # # # # # #
#     M E A S U R E M E N T S     #
# # # # # # # # # # # # # # # # # #

@dataclass
class Measurement(ABC):

    mdp: MDParams = field(repr=False)
    prop: Propagator = field(repr=False)
    model: Potential = field(repr=False)
    abscissa: str = ""
    ordinate: str = ""
    file_name: str = ""
    req_accur: float = np.nan
    typ_accur: str = "rel"
    observable: FloatArray = field(default_factory=lambda: np.array([]), 
                                   init=False, repr=False)

    def __post_init__(self):

        if self.abscissa is None:
            raise ValueError(f"abscissa missing in {type(self)}")

        if self.ordinate is None:
            raise ValueError(f"ordinate missing in {type(self)}")

        if not self.file_name:
            self.file_name = \
            f"{self.ordinate}.{self.abscissa}.{self.model.name}.{self.prop.name}.dat"

        self.observable = np.zeros(self.model.n_sample)

        if self.req_accur is not np.nan and self.model.n_sample == 1:
            raise ValueError("Target accuracy requires n_sample > 1")

        with open(self.file_name, "w") as f:
            f.write(f"# pid:  {os.getpid()}\n\n")
            f.write(f"# {str(self.model)}\n\n")
            f.write(f"# {str(self.mdp)}\n\n")
            f.write(f"# {str(self.prop)}\n\n")
            f.write(f"# {self.abscissa}\t\t")

    @abstractmethod
    def meas_init(self):
        pass

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def test_accuracy(self)->bool:

        mean, error = get_mean_and_error(self.observable)
        
        if self.typ_accur == "rel":
            if error > abs(mean) * self.req_accur:
                return False
        else:
            if error > self.req_accur:
                return False

        return(True)

    @abstractmethod
    def meas_out(self):
        pass

    def return_abscissa(self):

        if self.abscissa=="dt":
            return self.mdp.dt
        elif self.abscissa=="temp":
            return self.mdp.temp
        elif self.abscissa=="md_time":
            return self.mdp.r_time

        elif self.abscissa=="tau":
            if isinstance(self.prop,Langevin):
                return self.prop.tau
        elif self.abscissa=="dampg":
            if isinstance(self.prop,Brown):
                return self.prop.dampg

        elif self.abscissa=="f_pa":
            if isinstance(self.model,FKChain):
                return self.model.force_pa
            else:
                raise ValueError("force_pa not yet implemented in parent class")

        elif self.abscissa=="vel":
            if not self.prop.f_const_vel:
                raise ValueError("vel only allowed as abscissa if constrained")
            else:
                return self.prop.const_vel

        else:
            raise ValueError("Unknown abscissa.")

@dataclass
class MeasEnerg(Measurement):

    def __post_init__(self):
        self.ordinate = "nrg"
        super().__post_init__()
        with open(self.file_name, "a") as file:
            file.write(f"<v_pot_1> std_err\t")
            file.write(f"<v_pot_2> std_err")
            if hasattr(self.prop, "v"):
                file.write(f"\t<t_kin_1> std_err")
                file.write(f"\t<t_kin_2> std_err")
            file.write("\tn_observe\n")

    def meas_init(self):
        self.v_pot_1 = np.zeros(self.model.n_sample)
        self.v_pot_2 = np.zeros(self.model.n_sample)
        if hasattr(self.prop, "v"):
            self.t_kin_1 = np.zeros(self.model.n_sample)
            self.t_kin_2 = np.zeros(self.model.n_sample)

    def measure(self):

        i_time = self.mdp.i_time

        if i_time < 0 or self.mdp.i_run < 0:
            return

        if i_time % 10:
            return

        # "expensive" measurements

        config = self.prop.x, self.prop.f 	
        self.model.set_v_pot_mmt(config)
        self.v_pot_1 += self.model.v_pot_1
        self.v_pot_2 += self.model.v_pot_2

        if isinstance(self.prop, Langevin):

            self.prop.t_kin[:] = self.prop.mass * self.prop.v**2 / 2
            self.t_kin_1 += np.mean(self.prop.t_kin, axis=-1)
            self.t_kin_2 += np.mean(self.prop.t_kin**2, axis=-1)

    def test_accuracy(self)->bool:
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        if self.model.f_single_part == True:
            self.observable[:] = np.sum(self.v_pot_1, axis=-1) / n_observe
        else:
            self.observable[:] = self.v_pot_1 / n_observe
        converged = Measurement.test_accuracy(self)
        return(converged)

    def meas_out(self):
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        self.v_pot_1 /= n_observe
        self.v_pot_2 /= n_observe

        if hasattr(self.prop, "v"):
            self.t_kin_1 /= n_observe
            self.t_kin_2 /= n_observe

        with open(self.file_name, "a") as file:
            # first and second moment potential energy
            file.write(f"{self.return_abscissa():.6g}\t")
            mean_1, error = get_mean_and_error(self.v_pot_1)
            file.write(f"{mean_1:.7g} {error:.3g}\t")
            mean_2, error = get_mean_and_error(self.v_pot_2)
            file.write(f"{mean_2:.6g} {error:.3g}")
            # mean_1 and mean_2 could be further processed for config. specific heat

            # first and second moment kinetic energy
            if hasattr(self.prop, "v"):
                mean, error = get_mean_and_error(self.t_kin_1)
                file.write(f"\t{mean:.7g} {error:.3g}")
                mean, error = get_mean_and_error(self.t_kin_2)
                file.write(f"\t{mean:.6g} {error:.3g}")

            file.write(f"\t{self.mdp.n_observe}\n")


@dataclass
class MeasSpecHeat(MeasEnerg):

    def __post_init__(self):
        self.ordinate = "c_v"
        Measurement.__post_init__(self)
        with open(self.file_name, "a") as file:
            file.write(f"<c_p>\t std_err\tn_observe")
            file.write("\n")

    def meas_init(self):
        super().meas_init()

    def measure(self):
        super().measure()

    def test_accuracy(self)->bool:
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        if self.model.f_single_part:
            self.observable[:] = np.average(
                (self.v_pot_2/n_observe - (self.v_pot_1/n_observe)**2) / self.mdp.temp**2, axis=-1)
        else:
            self.observable[:] = \
                (self.v_pot_2/n_observe - (self.v_pot_1/n_observe)**2) / self.mdp.temp**2

        converged = Measurement.test_accuracy(self)
        return(converged)

    def meas_out(self):
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        spec_heat = (self.v_pot_2/n_observe - (self.v_pot_1/n_observe)**2) / self.mdp.temp**2
        with open(self.file_name, "a") as file:
            # first and second moment potential energy
            file.write(f"{self.return_abscissa():.6g}\t")
            mean_1, error = get_mean_and_error(spec_heat)
            file.write(f"{mean_1:.7g} {error:.3g}\t")
            file.write(f"\t{self.mdp.n_observe}\n")

@dataclass
class MeasMoments(Measurement):

    def __post_init__(self):
        self.ordinate = "mmt"
        super().__post_init__()
        with open(self.file_name, "a") as file:
            file.write(f"<x_2> std_err\t")
            file.write(f"<x_4> std_err\t")
            file.write(f"<x_6> std_err\t")
            file.write(f"<x_8> std_err\n")

    def meas_init(self):
        n_sample = self.prop.n_sample
        self.x_2 = np.zeros(n_sample)
        self.x_4 = np.zeros(n_sample)
        self.x_6 = np.zeros(n_sample)
        self.x_8 = np.zeros(n_sample)

    def measure(self):

        i_time = self.mdp.i_time

        if i_time < 0 or self.mdp.i_run < 0:
            return

        if i_time % 10:
            return

        x = self.prop.x
        self.x_2 += np.mean(x**2, axis=-1)
        self.x_4 += np.mean(x**4, axis=-1)
        self.x_6 += np.mean(x**6, axis=-1)
        self.x_8 += np.mean(x**8, axis=-1)

    def test_accuracy(self)->bool:
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        self.observable[:] = self.x_2 / n_observe
        converged = Measurement.test_accuracy(self)
        return(converged)


    def meas_out(self):

        n_observe = 1 + (self.mdp.n_observe-1) // 10
        self.x_2 /= n_observe
        self.x_4 /= n_observe
        self.x_6 /= n_observe
        self.x_8 /= n_observe

        with open(self.file_name, "a") as file:
            file.write(f"{self.return_abscissa():.6g}\t")
            mean, error = get_mean_and_error(self.x_2)
            file.write(f"{mean:.6g} {error:.3g}\t")
            mean, error = get_mean_and_error(self.x_4)
            file.write(f"{mean:.6g} {error:.3g}\t")
            mean, error = get_mean_and_error(self.x_6)
            file.write(f"{mean:.6g} {error:.3g}\t")
            mean, error = get_mean_and_error(self.x_8)
            file.write(f"{mean:.6g} {error:.3g}\n")

@dataclass
class MeasDiffusion(Measurement):

    def __post_init__(self):
        self.ordinate = "dff"
        super().__post_init__()
        with open(self.file_name, "a") as file:
            file.write("<{dx(t)-dx(0)}^2> std_err\t")
            file.write("<{com(t)-com(0)}^2> std_err\t")

    def meas_init(self):
        pass
        n_sample = self.prop.n_sample
        self.x_2_rel = np.zeros(n_sample)
        self.x_2_com = np.zeros(n_sample)

    def measure(self):

        i_time = self.mdp.i_time

        if i_time < 0 or self.mdp.i_run < 0:
            return

        if i_time == 0:
            print("making x_start_com")
            self.x_start_com = np.mean(self.prop.x, axis=-1)
            self.x_start_mono = self.prop.x - self.x_start_com[:, np.newaxis]

        self.x_now_com = np.mean(self.prop.x, axis=-1)
        self.x_now_mono = self.prop.x - self.x_now_com[:, np.newaxis]

        x_com_diff = (self.x_now_com-self.x_start_com)**2
        x_mono_diff = np.mean( (self.x_now_mono - self.x_start_mono)**2, axis=-1 )

        with open(self.file_name, "a") as file:
            file.write(f"{self.return_abscissa():.6g}\t")
            mean, error = get_mean_and_error(x_mono_diff)
            file.write(f"{mean:.6g} {error:.3g}\t")
            mean, error = get_mean_and_error(x_com_diff)
            file.write(f"{mean:.6g} {error:.3g}\n")


    def test_accuracy(self)->bool:
        return(True)


    def meas_out(self):
        pass


@dataclass
class MeasVelocity(Measurement):

    def __post_init__(self):
        self.ordinate = "vel"
        super().__post_init__()
        self.orig_pos = np.zeros(self.prop.x.shape[0])
        self.final_pos = np.zeros_like(self.orig_pos)
        self.observable = np.zeros_like(self.orig_pos)
        with open(self.file_name, "a") as file:
            file.write(f"velocity std_err\t")
            file.write(f"n_observe\n")

    def meas_init(self):
        pass

    def measure(self):
        i_time = self.mdp.i_time

        if i_time == 0:
            self.orig_pos[:] = np.average(self.prop.x, axis=-1)
            return

        if i_time == self.mdp.n_observe-1:
            self.final_pos[:] = np.average(self.prop.x, axis=-1)
            return


    def test_accuracy(self)->bool:
        elapsed_time = self.mdp.n_observe * self.mdp.dt
        self.observable[:] = (self.final_pos - self.orig_pos) / elapsed_time
        converged = Measurement.test_accuracy(self)
        return(converged)


    def meas_out(self):

        with open(self.file_name, "a") as file:
            abscissa = self.return_abscissa()
            mean, error = get_mean_and_error(self.observable)
            file.write(f"{abscissa:.6g}\t")
            file.write(f"{mean:.6g} {error:.3g}\t")
            file.write(f"{self.mdp.n_observe}\n")

@dataclass
class MeasForce(Measurement):

    def __post_init__(self):
        self.ordinate = "frc"
        super().__post_init__()
        with open(self.file_name, "a") as file:
            file.write(f"force std_err\t")
            file.write(f"n_observe\n")

    def meas_init(self):
        self.force_1 = np.zeros(self.prop.n_sample)
        self.observable = np.zeros(self.prop.n_sample)

    def measure(self):
        i_time = self.mdp.i_time

        if i_time < 0 or self.mdp.i_run < 0:
            return

        if i_time % 10:
            return

        # change: check into the question why self.prop.const_idx is different
        if isinstance(self.prop.const_idx, np.ndarray):
            self.force_1 += np.average(self.prop.f[:,self.prop.const_idx], axis=-1)
        else:
            self.force_1 += np.average(self.prop.f, axis=-1)


    def test_accuracy(self)->bool:
        n_observe = 1 + (self.mdp.n_observe-1) // 10
        self.observable[:] = self.force_1 / n_observe
        converged = Measurement.test_accuracy(self)
        return(converged)


    def meas_out(self):

        with open(self.file_name, "a") as file:
            abscissa = self.return_abscissa()
            n_observe = 1 + (self.mdp.n_observe-1) // 10
            self.observable[:] = self.force_1 / n_observe
            mean, error = get_mean_and_error(self.observable)
            file.write(f"{abscissa:.6g}\t")
            file.write(f"{-mean:.6g} {error:.3g}\t")
            file.write(f"{self.mdp.n_observe}\n")


# # # # # # # # # # # # # # # # # # # # # # #
#     U T I L I T Y   F U N C T I O N S     #
# # # # # # # # # # # # # # # # # # # # # # #

def set_random(g: FloatArray, f_rand: str) -> None:
    if f_rand=="unif":
        g[:] = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=g.shape)
    else:
        g[:] = np.random.normal(0.0, 1.0, size=g.shape)
        if f_rand=="cstr":
           g -= np.mean(g, axis=-1, keepdims=True)


def prepare_directory():

    folder_name = "Results"

    if not os.path.exists(folder_name):
        # Creates Results directory if it does not exist.
        os.makedirs(folder_name)
    else:
        # Delete all *.dat in Results
        for file_path in glob.glob(os.path.join(folder_name, "*.dat")):
            os.remove(file_path)

    with open("params.out", "w") as file:
        file.write("")

def write_info_to_params_out(mdp:MDParams, prop:Propagator, model:Potential):
    with open("params.out", "a") as file:
        file.write(f"# pid:  {os.getpid()}\n")
        file.write(f"# {str(mdp)}\n")
        file.write(f"# {str(prop)}\n")
        file.write(f"# {str(model)}\n")

@njit
def get_mean_and_error(array):
    n_data = array.shape[0]
    mean = np.mean(array) 
    if n_data==1: 
        std = np.nan
        return mean, std
    std = np.sqrt( (np.mean(array**2) - mean**2) / (n_data-1) )
    return mean, std

def round_down_to_n_sig_digits(x, n):
    if x == 0:
        return 0
    exponent = np.floor(np.log10(x))
    factor = 10 ** (exponent - n + 1)
    return np.floor(x / factor) * factor

# # # # # # # # # # # # # # # # # # #
#     M D   I N C L   F C T N S     #
# # # # # # # # # # # # # # # # # # #

def measure(mdp, meas_c):
    for meas in meas_c:
        meas.measure()
        if mdp.i_time == mdp.n_observe - 1:
            if meas.req_accur is np.nan:
                continue

            converged = meas.test_accuracy()

            if not converged:
                mdp.n_observe += max([1, int(mdp.n_observe/4)])
                mdp.n_observe = min(mdp.n_observe, mdp.n_obs_max)
                break

def shift_pos_back(prop, model):
     if isinstance(model, FKChain):  ## FK model
        n_shift_latt = (np.mean(prop.x, axis=1, keepdims=True) // model.latt_const).astype(int)
        prop.x -= n_shift_latt * model.latt_const
        prop.x += (prop.n_atom//2) * model.latt_const
            

def md(mdp:MDParams, prop:Propagator, model:Potential, meas_c:list[Measurement]):

    config = prop.x, prop.f

    prop.init_coeffs()

    if isinstance(prop, BPCLangevin):
        print("running", len(meas_c))
        print(meas_c[0].prop)
        print(meas_c[1].prop)

    for mdp.i_run in range(-mdp.n_pre_run, mdp.n_run):

        for meas in meas_c:
            meas.meas_init()

        mdp.i_time = -mdp.n_relax
        while mdp.i_time < mdp.n_observe:
            mdp.r_time = mdp.i_time * mdp.dt

            if mdp.i_time == 0:
                shift_pos_back(prop,model)

            model.set_force(config)
            prop.propagate()
            measure(mdp, meas_c)

            mdp.i_time += 1

        for meas in meas_c:
            meas.meas_out()

        if mdp.n_observe == mdp.n_obs_max:
            break


# # # # # # # # # #
#     M A I N     #
# # # # # # # # # #

def main():
    """
    This is an example wrapper file to run a simulation
    """
    n_sample, n_atom = 256, 128
    mdp = MDParams(dt = 2*np.pi/20, n_relax=1_000, n_observe=4_000)
    prop = GJLangevin(n_sample=n_sample, n_atom=n_atom, mdp=mdp)
    model = HarmOsc(n_sample=n_sample, n_atom=n_atom, box_x=mdp.box_x)
    meas1 = MeasEnerg(abscissa="dt", mdp=mdp, prop=prop, model=model)
    meas2 = MeasSpecHeat(abscissa="dt", mdp=mdp, prop=prop, model=model)

    # Prepare directory
    prepare_directory()
    write_info_to_params_out(mdp=mdp, prop=prop, model=model)

    # Run simulation
    md(mdp, prop, model, [meas1, meas2])

if __name__ == "__main__":
    main()
