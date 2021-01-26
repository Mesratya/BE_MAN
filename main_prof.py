from ship_params import *
from mmg_3dof import *
import matplotlib.pyplot as plt
import numpy as np

# Load ship model
ship_params = kvlcc2

# Set maneuver
maneuver_params = {'type': 'zig_zag', 'params': 10.}
#maneuver_params = {'type': 'turning_circle', 'params': 35.}

# Init MMG-3DOF
ship = Ship(ship_params, maneuver_params)
dyn_params = {'t_0': 0.,
              't_max': 1500.,
              'd_t': 1.,
              'dyn_system': ship}
dyn_ship_solver = DynSolver(dyn_params)

# MMG validations
print(ship.mat_M_inv)
# [[ 2.90315877e-09  0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  1.78555490e-09 -1.91303193e-12]
#  [-0.00000000e+00 -1.91303193e-12  3.03268796e-13]]

print(ship.hull.compute_T([2., -0.8, 7e-03]))
# [-2.73227239e+04  8.00908758e+06 -3.92100638e+08]

print(ship.propeller.compute_T([2., -0.8, 7e-03]))
# [4506388.85869666       0.               0.        ]
print(ship.rudder.compute_T([2., -0.8, 7e-03]))
# [ 0.00000000e+00  2.07450714e+06 -3.26238005e+08]

print(ship.diff_eq(0., [250., 680., 11., 2., -0.8, 7e-03]))
# [-7.91140769e-01 -2.00352097e+00  7.00000000e-03  4.32347317e-03 1.08637367e-02 -2.43016228e-04]

# Solve MMG-3DOF
[t, ship_eta, ship_mu, rudder_delta] = dyn_ship_solver.solve_dyn()

# Plot results

if maneuver_params['type'] == 'turning_circle':
    plt.figure("U")
    plt.grid('on')
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'],
             (ship_mu[:, 0] ** 2. + ship_mu[:, 1] ** 2.) ** 0.5 / ship.ship_params['mu_0'][0], 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("U / u_0 (-)")
    data = np.loadtxt('data_exp\\U_TC35.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("r")
    plt.grid('on')
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'],
             ship_mu[:, 2] * ship.ship_params['geo']['L_pp'] / ship.ship_params['mu_0'][0], 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("r * L_pp / u_0 (-)")
    data = np.loadtxt('data_exp\\r_TC35.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("Drift angle")
    plt.grid('on')
    beta = np.arctan(-ship_mu[:, 1] / ship_mu[:, 0])
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'], np.rad2deg(beta), 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("beta (deg)")
    data = np.loadtxt('data_exp\\beta_TC35.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("rudder angle")
    plt.grid('on')
    plt.plot(t, np.rad2deg(rudder_delta), 'b-')
    plt.xlabel("t")
    plt.ylabel("delta (deg)")



    plt.figure("trajec")
    plt.grid('on')
    x, y = ship_eta[:, 0], ship_eta[:, 1]
    plt.plot(y / ship.ship_params['geo']['L_pp'], x / ship.ship_params['geo']['L_pp'], 'b-')
    plt.xlabel("y  (-)")
    plt.ylabel("x (-)")
    data = np.loadtxt('data_exp\\traj_TC35.dat')
    plt.plot(data[:, 0], data[:, 1], 'r+')


    plt.show()


if maneuver_params['type'] == 'zig_zag':
    plt.figure("U")
    plt.grid('on')
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'],
             (ship_mu[:, 0] ** 2. + ship_mu[:, 1] ** 2.) ** 0.5 / ship.ship_params['mu_0'][0], 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("U / u_0 (-)")
    data = np.loadtxt('data_exp\\U_ZZ10.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("r")
    plt.grid('on')
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'],
             ship_mu[:, 2] * ship.ship_params['geo']['L_pp'] / ship.ship_params['mu_0'][0], 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("r * L_pp / u_0 (-)")
    data = np.loadtxt('data_exp\\r_ZZ10.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("Drift angle")
    plt.grid('on')
    beta = np.arctan(-ship_mu[:, 1] / ship_mu[:, 0])
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'], np.rad2deg(beta), 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("beta (deg)")
    data = np.loadtxt('data_exp\\beta_ZZ10.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("rudder angle")
    plt.grid('on')
    plt.plot(t, np.rad2deg(rudder_delta), 'b-')
    plt.xlabel("t")
    plt.ylabel("delta (deg)")

    plt.figure("heading angle")
    plt.grid('on')
    psi = ship_eta[:,2]
    plt.plot(t * ship.ship_params['mu_0'][0] / ship.ship_params['geo']['L_pp'], np.rad2deg(psi), 'b-')
    plt.xlabel("t * u_0 / L_pp (-)")
    plt.ylabel("psi (deg)")
    data = np.loadtxt('data_exp\\psi_ZZ10.dat')
    plt.plot(data[:, 0], data[:, 1], 'r--')

    plt.figure("trajec")
    plt.grid('on')
    x,y = ship_eta[:,0],ship_eta[:,1]
    plt.plot(y / ship.ship_params['geo']['L_pp'], x / ship.ship_params['geo']['L_pp'], 'b-')
    plt.xlabel("y  (-)")
    plt.ylabel("x (-)")
    data = np.loadtxt('data_exp\\traj_ZZ10.dat')
    plt.plot(data[:, 0] , data[:, 1] , 'r--')

    plt.show()
