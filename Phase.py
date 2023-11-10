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
        self.p = np.ones(N)*p0
        self.rho = np.ones(N)*rho0
        self.u = mdot0/rho0
        self.mdot = mdot0
        self.dudx = np.zeros(N)

        self.cp_func = cp_func
        self.rho_func = rho_func
        self.kappa_func = kappa_func

    # def cp(self, T, p):
    #     return self.cp_func(T, p)
