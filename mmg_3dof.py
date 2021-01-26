from sys import exit
import numpy as np
from scipy import interpolate as interp


class DynSolver:
    def __init__(self, dyn_params):
        self.t = dyn_params['t_0']
        self.t_max = dyn_params['t_max']
        self.d_t = dyn_params['d_t']
        self.dyn_system = dyn_params['dyn_system']

    def solve_dyn(self):
        eta = []
        mu = []
        t = []
        control = []

        while self.t < self.t_max:
            self.dyn_system.controler(self.d_t)

            [vect_eta_mu, self.t] = self.solve_rk4(self.t, self.d_t,
                                                   np.concatenate((self.dyn_system.eta, self.dyn_system.mu)))
            self.dyn_system.eta = vect_eta_mu[:3]
            self.dyn_system.mu = vect_eta_mu[3:]

            eta.append(self.dyn_system.eta)
            mu.append(self.dyn_system.mu)
            t.append(self.t)
            control.append(self.dyn_system.return_control())

        return [np.asanyarray(t), np.asanyarray(eta), np.asanyarray(mu), np.asanyarray(control)]

    def solve_rk4(self, t, d_t, y):

        dy1 = self.dyn_system.diff_eq(t, y)
        dy2 = self.dyn_system.diff_eq(t + 0.5 * d_t, y + 0.5 * d_t * dy1)
        dy3 = self.dyn_system.diff_eq(t + 0.5 * d_t, y + 0.5 * d_t * dy2)
        dy4 = self.dyn_system.diff_eq(t + d_t, y + d_t * dy3)

        y_sol = y + d_t * (dy1 + 2. * dy2 + 2. * dy3 + dy4) / 6.
        t_sol = t + d_t
        return [y_sol, t_sol]


class Ship:

    def __init__(self, ship_params, maneuver_params):
        # Load ship parameters
        self.ship_params = ship_params

        # Init hull, propeller and rudder MMG components
        self.hull = Hull(ship_params)
        self.propeller = Propeller(ship_params)
        self.rudder = Rudder(ship_params)
        # self.wind = Wind(ship_params)  # To be used in Section 3.2 of the lab

        # Mass matrix inversion
        ############### To be implemented ###############
        m = self.ship_params['rho'] * self.ship_params['geo']['volume']
        mx = self.ship_params['added_mass']['mp_x'] * 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] **2 ) * self.ship_params['geo']['d']
        my = self.ship_params['added_mass']['mp_y'] * 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] **2 ) * self.ship_params['geo']['d']
        Jz = self.ship_params['added_mass']['Jp_z'] * 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] **4 ) * self.ship_params['geo']['d']
        xg = self.ship_params['geo']['x_g']
        Izg = m * ( (0.25*self.ship_params['geo']['L_pp']) ** 2 )


        mat_M = np.array([[m+mx,0,0],[0,m+my,xg*m],[0,xg*m,Izg+(xg**2)*m+Jz]])

        ############### To be implemented ###############
        self.mat_M_inv = np.linalg.inv(mat_M)

        # Ship velocities and positions
        self.mu = np.array(self.ship_params['mu_0'])
        self.eta = np.zeros(3)

        # Maneuver setup
        self.maneuver = {'type': maneuver_params['type'], 'params': np.deg2rad(maneuver_params['params'])}

    def diff_eq(self, t, vect_eta_mu):
        eta = vect_eta_mu[:3]
        mu = vect_eta_mu[3:]

        mat_rot_psi = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0.],
                                [np.sin(eta[2]), np.cos(eta[2]), 0.],
                                [0., 0., 1.]])
        eta_p = mat_rot_psi.dot(mu)

        ############### To be implemented ###############
        #2.11
        T = self.hull.compute_T(mu) + self.propeller.compute_T(mu) + self.rudder.compute_T(mu)

        m = self.ship_params['rho'] * self.ship_params['geo']['volume']
        mx = self.ship_params['added_mass']['mp_x'] * 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] ** 2) * \
             self.ship_params['geo']['d']
        my = self.ship_params['added_mass']['mp_y'] * 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] ** 2) * \
             self.ship_params['geo']['d']
        xg = self.ship_params['geo']['x_g']
        r = mu[2]

        C = np.array([[0,-(m+my)*r,-xg*m*r],[(m+mx)*r,0,0],[xg*m*r,0,0]])
        mu_p = self.mat_M_inv.dot( T - C.dot(mu) )


        ############### To be implemented ###############

        return np.concatenate((eta_p, mu_p))

    def controler(self, d_t):
        if self.maneuver['type'] == 'turning_circle':
            delta = self.maneuver['params']

        elif self.maneuver['type'] == 'zig_zag':
            if np.abs(self.eta[2]) > np.abs(self.maneuver['params']) and \
                    np.sign(self.eta[2]) == np.sign(self.maneuver['params']):
                self.maneuver['params'] = -self.maneuver['params']
            delta = self.maneuver['params']

        else:
            exit("Error: 'type' for maneuver?")

        self.rudder.set_delta(delta, d_t)

    def return_control(self):
        return self.rudder.delta


class Propeller:

    def __init__(self, ship_params):
        self.ship_params = ship_params
        self.T = np.zeros(3)

    def compute_T(self, mu):

        u = mu[0]
        beta = np.arctan(-mu[1] / mu[0])
        U = (mu[0] ** 2 + mu[1] ** 2) ** 0.5
        rp = mu[2] * self.ship_params['geo']['L_pp'] / U # adimensional yaw rate r'

        beta_p = beta - self.ship_params['propeller']['xp_p'] * rp
        w_p = self.ship_params['propeller']['w_p0'] * np.exp(-4. * beta_p ** 2.)

        Jp = u * (1 - w_p) / (self.ship_params['propeller']['n_p'] * self.ship_params['propeller']['D_p'])
        Kt = self.ship_params['propeller']['k_0'] + self.ship_params['propeller']['k_1'] * Jp + self.ship_params['propeller']['k_2'] * (Jp ** 2)

        Tp = self.ship_params['rho'] * (self.ship_params['propeller']['n_p'] ** 2) * (self.ship_params['propeller']['D_p'] ** 4) * Kt

        X_p = (1 - self.ship_params['propeller']['t_p']) * Tp
        Y_p = 0
        N_p = 0

        self.T = np.array([X_p, Y_p, N_p])
        return self.T



class Hull:

    def __init__(self, ship_params):
        self.ship_params = ship_params
        self.T = np.zeros(3)

    def compute_T(self, mu):

        U = (mu[0] ** 2 + mu[1] ** 2) ** 0.5
        rp = mu[2] * self.ship_params['geo']['L_pp'] / U  # adimensional yaw rate r'
        vm_p = mu[1] / U  # adim lateral velocity at midship


        X_hp = - self.ship_params['hull_der']['Xp_0'] + self.ship_params['hull_der']['Xp_vv'] * (vm_p**2) + self.ship_params['hull_der']['Xp_vr']*vm_p*rp + self.ship_params['hull_der']['Xp_rr'] * (rp**2) + self.ship_params['hull_der']['Xp_vvvv'] * (vm_p**4)
        Y_hp = self.ship_params['hull_der']['Yp_v']*vm_p + self.ship_params['hull_der']['Yp_r']*rp + self.ship_params['hull_der']['Yp_vvv']*(vm_p**3) + self.ship_params['hull_der']['Yp_vvr']*(vm_p**2)*rp + self.ship_params['hull_der']['Yp_vrr'] * vm_p * (rp**2) + self.ship_params['hull_der']['Yp_rrr'] * (rp**3)
        N_hp = self.ship_params['hull_der']['Np_v']*vm_p + self.ship_params['hull_der']['Np_r']*rp + self.ship_params['hull_der']['Np_vvv']*(vm_p**3) + self.ship_params['hull_der']['Np_vvr']*(vm_p**2)*rp + self.ship_params['hull_der']['Np_vrr'] * vm_p * (rp**2) + self.ship_params['hull_der']['Np_rrr'] * (rp**3)


        X_h = 0.5 * self.ship_params['rho'] * self.ship_params['geo']['L_pp'] * self.ship_params['geo']['d'] * (U**2) * X_hp
        Y_h = 0.5 * self.ship_params['rho'] * self.ship_params['geo']['L_pp'] * self.ship_params['geo']['d'] * (U**2) * Y_hp
        N_h = 0.5 * self.ship_params['rho'] * (self.ship_params['geo']['L_pp'] ** 2) * self.ship_params['geo']['d'] * (U**2) * N_hp

        self.T = np.array([X_h, Y_h, N_h])


        return self.T


class Rudder:

    def __init__(self, ship_params):
        self.ship_params = ship_params
        self.T = np.zeros(3)
        self.delta = 0.

    def set_delta(self, delta, d_t):
        rr_set = (delta - self.delta) / d_t
        if np.abs(rr_set) > self.ship_params['rudder']['rr_max']:
            self.delta += np.sign(rr_set) * self.ship_params['rudder']['rr_max'] * d_t
        else:
            self.delta = delta

    def compute_T(self, mu):
        U = (mu[0] ** 2 + mu[1] ** 2) ** 0.5
        beta = np.arctan(-mu[1] / mu[0])
        rp = mu[2] * self.ship_params['geo']['L_pp'] / U
        beta_p = beta - self.ship_params['propeller']['xp_p'] * rp
        w_p = self.ship_params['propeller']['w_p0'] * np.exp(-4. * beta_p ** 2.)
        u_p = mu[0] * (1. - w_p)

        J_p = u_p / (self.ship_params['propeller']['n_p'] * self.ship_params['propeller']['D_p'])
        K_t = self.ship_params['propeller']['k_0'] + \
              self.ship_params['propeller']['k_1'] * J_p + \
              self.ship_params['propeller']['k_2'] * J_p ** 2.

        eta = self.ship_params['propeller']['D_p'] / self.ship_params['rudder']['H_r']
        tmp_u_r = 1 + self.ship_params['rudder']['kappa'] * ((1 + 8 * K_t / (np.pi * J_p ** 2.)) ** 0.5 - 1)
        u_r = self.ship_params['rudder']['epsilon'] * u_p * (eta * tmp_u_r ** 2. + (1. - eta)) ** 0.5

        beta_r = beta - self.ship_params['rudder']['lp_r'] * rp
        if beta_r < 0.:
            gamma_r = self.ship_params['rudder']['gamma_r'][0]
        else:
            gamma_r = self.ship_params['rudder']['gamma_r'][1]
        v_r = U * gamma_r * beta_r

        alpha_r = self.delta - np.arctan(v_r / u_r)
        U_r = (u_r ** 2. + v_r ** 2.) ** 0.5
        if 'lambda_r' in self.ship_params['rudder']:
            lambda_r = self.ship_params['rudder']['lambda_r']
        else:
            lambda_r = self.ship_params['rudder']['H_r'] ** 2. / self.ship_params['rudder']['A_r']
        f_alpha = 6.13 * lambda_r / (lambda_r + 2.25)

        F_n = 0.5 * self.ship_params['rho'] * self.ship_params['rudder']['A_r'] * U_r ** 2. * f_alpha * np.sin(alpha_r)
        x_r = self.ship_params['rudder']['xp_r'] * self.ship_params['geo']['L_pp']
        x_h = self.ship_params['rudder']['xp_h'] * self.ship_params['geo']['L_pp']

        X_r = -(1 - self.ship_params['rudder']['t_r']) * F_n * np.sin(self.delta)
        Y_r = -(1 + self.ship_params['rudder']['a_h']) * F_n * np.cos(self.delta)
        N_r = -(x_r + self.ship_params['rudder']['a_h'] * x_h) * F_n * np.cos(self.delta)

        self.T = np.array([X_r, Y_r, N_r])
        return self.T


class Wind:

    def __init__(self, ship_params):
        self.ship_params = ship_params
        self.T = np.zeros(3)
        [self.vect_C_X, self.vect_C_Y, self.vect_C_N] = self.init_vect_C()
        [self.vect_A, self.vect_B, self.vect_C] = self.init_vect_A_B_C()

    def init_vect_C(self):
        B = self.ship_params['geo']['B']
        L_pp = self.ship_params['geo']['L_pp']
        A_SS = self.ship_params['ship_wind']['A_L']
        A_L = self.ship_params['ship_wind']['A_SS']
        A_T = self.ship_params['ship_wind']['A_T']
        S = self.ship_params['ship_wind']['S']
        C = self.ship_params['ship_wind']['C']
        M = self.ship_params['ship_wind']['M']
        vect_C_X = np.array([1., 2. * A_L / L_pp ** 2., 2. * A_T / B ** 2., L_pp / B, S / L_pp, C / L_pp, M])
        vect_C_Y = np.array([1., 2. * A_L / L_pp ** 2., 2. * A_T / B ** 2., L_pp / B, S / L_pp, C / L_pp, A_SS / A_L])
        vect_C_N = np.array([1., 2. * A_L / L_pp ** 2., 2. * A_T / B ** 2., L_pp / B, S / L_pp, C / L_pp])
        return [vect_C_X, vect_C_Y, vect_C_N]

    def init_vect_A_B_C(self):
        mat_A = np.array([
            [2.152, -5.00, 0.243, -0.164, 0, 0, 0],
            [1.714, -3.33, 0.145, -0.121, 0, 0, 0],
            [1.818, -3.97, 0.211, -0.143, 0, 0, 0.033],
            [1.965, -4.81, 0.243, -0.154, 0, 0, 0.041],
            [2.333, -5.99, 0.247, -0.190, 0, 0, 0.042],
            [1.726, -6.54, 0.189, -0.173, 0.348, 0, 0.048],
            [0.913, -4.68, 0, -0.104, 0.482, 0, 0.052],
            [0.457, -2.88, 0, -0.068, 0.346, 0, 0.053],
            [0.341, -0.91, 0, -0.031, 0, 0, 0.032],
            [0.355, 0, 0, 0, -0.247, 0, 0.018],
            [0.601, 0, 0, 0, -0.372, 0, -0.02],
            [0.651, 1.29, 0, 0, -0.582, 0, -0.031],
            [0.564, 2.54, 0, 0, -0.748, 0, -0.024],
            [-0.142, 3.58, 0, 0.047, -0.700, 0, -0.028],
            [-0.677, 3.64, 0, 0.069, -0.529, 0, -0.032],
            [-0.723, 3.14, 0, 0.064, -0.475, 0, -0.032],
            [-2.148, 2.56, 0, 0.081, 0, 1.27, -0.027],
            [-2.707, 3.97, -0.175, 0.126, 0, 1.81, 0],
            [-2.529, 3.76, -0.174, 0.128, 0, 1.55, 0]
        ])

        mat_B = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0.096, 0.22, 0, 0, 0, 0, 0],
            [0.176, 0.71, 0, 0, 0, 0, 0],
            [0.225, 1.38, 0, 0.023, 0, -0.29, 0],
            [0.329, 1.82, 0, 0.043, 0, -0.59, 0],
            [1.164, 1.26, 0.121, 0, -0.242, -0.95, 0],
            [1.163, 0.96, 0.101, 0, -0.177, -0.88, 0],
            [0.916, 0.53, 0.069, 0, 0, -0.65, 0],
            [0.844, 0.55, 0.082, 0, 0, -0.54, 0],
            [0.889, 0, 0.138, 0, 0, -0.66, 0],
            [0.799, 0, 0.155, 0, 0, -0.55, 0],
            [0.797, 0, 0.151, 0, 0, -0.55, 0],
            [0.996, 0, 0.184, 0, -0.212, -0.66, 0.34],
            [1.014, 0, 0.191, 0, -0.28, -0.69, 0.44],
            [0.784, 0, 0.166, 0, -0.209, -0.53, 0.38],
            [0.536, 0, 0.176, -0.029, -0.163, 0, 0.27],
            [0.251, 0, 0.106, -0.022, 0, 0, 0],
            [0.125, 0, 0.046, -0.012, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        mat_C = np.array([
            [0, 0, 0, 0, 0, 0],
            [0.0596, 0.061, 0, 0, 0, -0.074],
            [0.1106, 0.204, 0, 0, 0, -0.17],
            [0.2258, 0.245, 0, 0, 0, -0.38],
            [0.2017, 0.457, 0, 0.0067, 0, -0.472],
            [0.1759, 0.573, 0, 0.0118, 0, -0.523],
            [0.1925, 0.48, 0, 0.0115, 0, -0.546],
            [0.2133, 0.315, 0, 0.0081, 0, -0.526],
            [0.1827, 0.254, 0, 0.0053, 0, -0.443],
            [0.2627, 0, 0, 0, 0, -0.508],
            [0.2102, 0, -0.0195, 0, 0.0335, -0.492],
            [0.1567, 0, -0.0258, 0, 0.0497, -0.457],
            [0.0801, 0, -0.0311, 0, 0.074, -0.396],
            [-0.0189, 0, -0.0488, 0.0101, 0.1128, -0.42],
            [0.0256, 0, -0.0422, 0.01, 0.0889, -0.463],
            [0.0552, 0, -0.0381, 0.0109, 0.0689, -0.476],
            [0.0881, 0, -0.0306, 0.0091, 0.0366, -0.415],
            [0.0851, 0, -0.0122, 0, 0.0025, -0.22],
            [0, 0, 0, 0, 0, 0]
        ])

        gamma_r = np.deg2rad(np.linspace(0., 180., 19))
        vect_A = interp.interp1d(gamma_r, mat_A.T)
        vect_B = interp.interp1d(gamma_r, mat_B.T)
        vect_C = interp.interp1d(gamma_r, mat_C.T)
        return [vect_A, vect_B, vect_C]

    def compute_T(self, eta, mu):
        tws = self.ship_params['ship_wind']['vel_wind']
        twa = np.radians(self.ship_params['ship_wind']['psi_wind']) - eta[2]
        bs = (mu[0] ** 2 + mu[1] ** 2) ** 0.5

        U_r = (bs ** 2. + tws ** 2. + 2. * bs * tws * np.cos(twa)) ** 0.5
        if U_r != 0.:
            gamma_r = np.arctan(tws * np.sin(twa) / (tws * np.cos(twa) + bs))
        else:
            gamma_r = 0.

        sign_gamma_r = np.sign(gamma_r)
        C_X = -self.vect_A(gamma_r * sign_gamma_r).dot(self.vect_C_X)
        C_Y = -self.vect_B(gamma_r * sign_gamma_r).dot(self.vect_C_Y) * sign_gamma_r
        C_N = -self.vect_C(gamma_r * sign_gamma_r).dot(self.vect_C_N) * sign_gamma_r

        X_w = 0.5 * self.ship_params['ship_wind']['rho'] * self.ship_params['ship_wind']['A_T'] * C_X
        Y_w = 0.5 * self.ship_params['ship_wind']['rho'] * self.ship_params['ship_wind']['A_L'] * C_Y
        N_w = 0.5 * self.ship_params['ship_wind']['rho'] * self.ship_params['ship_wind']['A_L'] * C_N * \
              self.ship_params['geo']['L_pp']

        self.T = np.array([X_w, Y_w, N_w]) * U_r ** 2.
        return self.T
