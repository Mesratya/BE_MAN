kvlcc2 = {
    'rho': 1025.,

    'mu_0': [15.5 * 0.514, 0., 0.],

    'geo': {'L_pp': 320.,
            'B': 58.,
            'd': 20.8,
            'volume': 312622,
            'x_g': 11.1,
            'C_b': 0.81},

    'hull_der': {'Xp_0': 0.022,
                 'Xp_vv': -0.04,
                 'Xp_vr': 0.002,
                 'Xp_rr': 0.011,
                 'Xp_vvvv': 0.771,
                 'Yp_v': -0.315,
                 'Yp_r': 0.083,
                 'Yp_vvv': -1.607,
                 'Yp_vvr': 0.379,
                 'Yp_vrr': -0.391,
                 'Yp_rrr': 0.008,
                 'Np_v': -0.137,
                 'Np_r': -0.049,
                 'Np_vvv': -0.03,
                 'Np_vvr': -0.294,
                 'Np_vrr': 0.055,
                 'Np_rrr': -0.013},

    'added_mass': {'mp_x': 0.022,
                   'mp_y': 0.223,
                   'Jp_z': 0.011},

    'propeller': {'D_p': 9.86,
                  't_p': 0.22,
                  'xp_p': -0.48,
                  'w_p0': 0.35,
                  'k_0': 0.293,
                  'k_1': -0.275,
                  'k_2': -0.139,
                  'n_p': 1.53},

    'rudder': {'H_r': 15.8,
               'A_r': 112.5,
               'xp_r': -0.5,
               'a_h': 0.312,
               't_r': 0.387,
               'xp_h': -0.464,
               'epsilon': 1.09,
               'kappa': 0.5,
               'lp_r': -0.71,
               'lambda_r': 1.827,
               'gamma_r': [0.395, 0.64],
               'rr_max': 0.0407},

    'ship_wind': {'A_L': 4910,
                  'A_T': 1624,
                  'A_SS': 750,
                  'S': 375,
                  'C': 160,
                  'M': 0,
                  'rho': 1.225,
                  'vel_wind': 0. * 15.5 * 0.514,
                  'psi_wind': -90.}
}