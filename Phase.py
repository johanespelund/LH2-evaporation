import numpy as np

T0 = 300
p0 = 1e5
rho0 = 1
mdot0 = 1e-4


class Phase:
    """
    Class for collecting all arrays for a phase.
    """
    def __init__(self, x_min, x_max, N, phase_flag,
                 cp_func, rho_func, kappa_func):
        self.phase_flag = phase_flag
        self.x = np.linspace(x_min, x_max, N)
        self.dx = np.gradient(self.x)
        self.T = np.ones(N)*T0
        self.p = p0
        self.calc_rho = np.ones(N)*rho0
        self.u = mdot0/rho0
        self.mdot = mdot0
        self.dudx = np.zeros(N)

        self.calc_cp = cp_func
        self.calc_rho = rho_func
        self.calc_kappa = kappa_func

    def set_mdot(self, mdot):
        self.mdot = mdot
    # def cp(self, T, p):
    #     return self.cp_func(T, p)
