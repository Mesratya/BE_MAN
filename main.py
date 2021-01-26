from ship_params import *
from mmg_3dof import *
import matplotlib.pyplot as plt
import numpy as np

# Load ship model
ship_params = kvlcc2

# Set maneuver
maneuver_params = {'type': 'zig_zag', 'params': 10.}
# maneuver_params = {'type': 'turning_circle', 'params': 35.}

# Init MMG-3DOF
ship = Ship(ship_params, maneuver_params)
dyn_params = {'t_0': 0.,
              't_max': 1500.,
              'd_t': 1.,
              'dyn_system': ship}
dyn_ship_solver = DynSolver(dyn_params)

# Solve MMG-3DOF
[t, ship_eta, ship_mu, rudder_delta] = dyn_ship_solver.solve_dyn()

# Plot results

Lpp = ship_params['geo']['L_pp']
u_0 = ship_params['mu_0'][0]

psi_dat = np.loadtxt('data_exp/psi_ZZ10.dat')
t_exp = psi_dat[:,0]
psi_exp = psi_dat[:,1]

t_sim = t * u_0 / Lpp
psi_sim = ship_eta[:,2]

plt.plot(t_exp,psi_exp,label = "exper")
plt.plot(t_sim,psi_sim,label = 'simulation')

plt.legend()
plt.show()