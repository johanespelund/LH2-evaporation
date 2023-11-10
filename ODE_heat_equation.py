import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])

from scipy.integrate import solve_bvp, solve_ivp
import numpy as np
from CoolProp.CoolProp import PropsSI  # Import PropsSI from CoolProp
from collections.abc import Iterable
from thermopack.cpa import cpa


water = cpa('H2O', 'SRK')
M = water.compmoleweight(1) * 1e-3  # [kg/mol]

from KGT import mass_flux_HKS, sigma_condensation
from NET import mass_flux_NET, heat_flux_gas_NET

# Parameters
p0 = 913  # Initial pressure
psat = 913.15
T0 = 273.15 + 6.1  # Temperature at x=0
Tliq = 273.15 + 5.7
TL = 273.15 + 40  # Temperature at x=L, assuming 60C for example
L_gas = 10e-3  # Length of the domain
L_liq = -5e-3
N = 50
mdot0 =  3.78e-4  # Mass flow rate
mdot0_HKS = mass_flux_HKS(sigma_condensation(Tliq, T0, 3), Tliq, T0, p0, p0, M)
# mdot = mdot_HKS
qdot = -10.59  # Heat flux
qliq = 433.2  # Heat flux\
    
Lqq = 0.236e-4
Lww = 24.77e-8
Lqw = -0.111

# mdot0 = mass_flux_NET(T0, Tliq, p0, psat, 8.314, M, Lww, Lqw)
# qgas = heat_flux_gas_NET(T0, Tliq, p0, psat, 8.314, M, Lqq, Lqw)

# print(mdot0, qgas)

class Phase:
    def __init__(self, x_min, x_max, phase_flag):
        self.phase_flag = phase_flag
        self.x = np.linspace(x_min, x_max, N)
        self.dx = np.gradient(self.x)
        self.T = np.ones(N)*T0
        self.p = np.ones(N)*p0
        self.rho = np.ones(N)*1
        self.u = mdot0/1
        self.mdot = mdot0
        self.dudx = np.zeros(N)


def calc_cp(T, p, phase_flag):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        cp = np.array([calc_cp(T_val, p_val, phase_flag) for T_val, p_val in zip(T, p)])
        return cp
    else:
        _, Cp_liq = water.enthalpy(T, p, [1], phase_flag, dhdt=True)
        return Cp_liq/M

# Define the function for rho(T)
def calc_rho(T, p, phase_flag):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        rho = np.array([calc_rho(T_val, p_val, phase_flag) for T_val, p_val in zip(T, p)])
        return rho
    else:
        specific_volume, = water.specific_volume(T, p, [1], phase_flag) # [m^3/mol]
        return M/specific_volume

# Define the function for kappa(T)
def calc_kappa(T, p, phase_flag):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        kappa = np.array([calc_kappa(T_val, p_val, phase_flag) for T_val, p_val in zip(T, p)])
        return kappa
    return kappa_vap if (phase_flag == water.VAPPH) else kappa_liq

# ODE system (in first-order form)
def dTdx(x, y, phase_flag):
    # T[0] is temperature, T[1] is its gradient
    T, dTdx = y
    _mdot = mdot0#*(1 - 200*xx)
    dudx = (_mdot * calc_cp(T, vap.p, phase_flag) / calc_kappa(T, liq.p, phase_flag)) * dTdx
    return np.vstack((dTdx, dudx))

# Boundary conditions
def bc_gas(ya, yb, phase_flag):
    kappa_inlet = calc_kappa(T0, p0, phase_flag)
    # return np.array([ya[0] - T0, yb[0] - TL])
    return np.array([ya[0] - T0, -kappa_inlet*ya[1] - qdot])

# Boundary conditions
def bc_liq(ya, yb, phase_flag):
    kappa_inlet = calc_kappa(T0, p0, phase_flag)
    # print(kappa_inlet, qliq)
    # return np.array([ya[0] - T0, yb[0] - TL])
    return np.array([yb[0] - Tliq, -kappa_inlet*yb[1] - qliq])

# Initial guess
y0 = np.ones((2, N))*T0

# Solve BVP
liq = Phase(L_liq, -0, water.LIQPH)
vap = Phase(0, L_gas, water.VAPPH)

for _ in range(20):
    sol_gas = solve_bvp(lambda x,y :dTdx(x, y, vap.phase_flag), lambda x,y :bc_gas(x, y, vap.phase_flag), vap.x, y0)
    vap.T = sol_gas.y[0]
    vap.rho = calc_rho(vap.T, vap.p, 1)
    vap.u = (vap.mdot/vap.rho)
    vap.p[1:] = (vap.p - vap.dx*vap.rho*np.gradient(vap.u, vap.x))[:-1]
    
    sol_liq = solve_bvp(lambda x,y :dTdx(x, y, liq.phase_flag), lambda x,y :bc_liq(x, y, liq.phase_flag), liq.x, y0)
    liq.T = sol_liq.y[0]
    liq.rho = calc_rho(liq.T, liq.p, 1)
    liq.u = (liq.mdot/liq.rho)
    liq.p[1:] = (liq.p - liq.dx*vap.rho*np.gradient(liq.u, vap.x))[:-1]

# Check if the solver converged
if sol_liq.status == 0 and sol_gas.status == 0:
    print('Success: The solver converged.')
else:
    print('Warning: The solver did not converge.')


from data_jafari import jafari_913Pa_40heat

plt.plot(vap.T - 273.15,vap.x * 1000)  # Convert x to mm
plt.plot(liq.T - 273.15, liq.x * 1000, 'C3')  # Convert x to mm
plt.plot(jafari_913Pa_40heat[:,0], jafari_913Pa_40heat[:,1], marker='>', lw=0, color='C2', label="Jafari et al. (913 Pa)")

plt.fill_between(np.linspace(5, 15), 0, L_liq*1000, alpha=0.5)
plt.xlabel('Temperature (deg. C)')
plt.ylabel('x [mm]')
plt.grid()
plt.legend()
# plt.xlim((-10, 20))
# plt.ylim((-10, 35))
plt.tight_layout()
plt.show()
