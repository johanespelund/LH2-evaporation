from data_jafari import jafari_913Pa_40heat
from NET import mass_flux_NET, heat_flux_gas_NET
import thermo
from KGT import mass_flux_HKS, sigma_condensation
from thermopack.cpa import cpa
from Phase import Phase
from CoolProp.CoolProp import PropsSI  # Import PropsSI from CoolProp
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])


# Parameters
p0 = 913  # Initial pressure
psat = 913.15
Tgas = 273.15 + 6.1  # Temperature at x=0
Tliq = 273.15 + 5.7
TL = 273.15 + 40  # Temperature at x=L, assuming 60C for example
L_gas = 10e-3  # Length of the domain
L_liq = -5e-3
N = 50
mdot0 = 3.78e-4  # Mass flow rate
# mdot0_HKS = mass_flux_HKS(sigma_condensation(Tliq, T0, 3), Tliq, T0, p0, p0, M)
# mdot = mdot_HKS
qgas = -10.59  # Heat flux
qliq = 433.2  # Heat flux\

Lqq = 0.236e-4
Lww = 24.77e-8
Lqw = -0.111

# mdot0 = mass_flux_NET(T0, Tliq, p0, psat, 8.314, M, Lww, Lqw)
# qgas = heat_flux_gas_NET(T0, Tliq, p0, psat, 8.314, M, Lqq, Lqw)

# print(mdot0, qgas)


# ODE system (in first-order form)

water = thermo.water
M = thermo.M

liq = Phase(L_liq, -0, N, water.LIQPH, 
            lambda T, p: thermo.calc_cp(T, p, water.LIQPH),
            lambda T, p: thermo.calc_rho(T, p, water.LIQPH),
            lambda T, p: thermo.calc_kappa(T, p, water.LIQPH))
vap = Phase(0, L_gas, N, water.VAPPH,
            lambda T, p: thermo.calc_cp(T, p, water.VAPPH),
            lambda T, p: thermo.calc_rho(T, p, water.VAPPH),
            lambda T, p: thermo.calc_kappa(T, p, water.VAPPH))

def dTdx(x, y, phase):
    # T[0] is temperature, T[1] is its gradient
    T, dTdx = y
    dudx = (mdot0 * phase.cp_func(T, vap.p) /
            phase.kappa_func(T, liq.p)) * dTdx
    return np.vstack((dTdx, dudx))

# Boundary conditions
def bc_gas(ya, yb, phase):
    kappa_inlet = phase.kappa_func(Tgas, p0)
    return np.array([ya[0] - Tgas, -kappa_inlet*ya[1] - qgas])

def bc_liq(ya, yb, phase):
    kappa_inlet = phase.kappa_func(Tliq, p0)
    return np.array([yb[0] - Tliq, ya[0] - (Tliq + 2)])

# Initial guess
y0 = np.ones((2, N))*Tgas

# Solve BVP

for _ in range(10):
    sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap),
                        lambda x, y: bc_gas(x, y, vap), vap.x, y0)
    vap.T = sol_gas.y[0]
    vap.rho = thermo.calc_rho(vap.T, vap.p, 1)
    vap.u = (vap.mdot/vap.rho)
    vap.p[1:] = (vap.p - vap.dx*vap.rho*np.gradient(vap.u, vap.x))[:-1]

    sol_liq = solve_bvp(lambda x, y: dTdx(x, y, liq),
                        lambda x, y: bc_liq(x, y, liq), liq.x, y0)
    liq.T = sol_liq.y[0]
    liq.rho = thermo.calc_rho(liq.T, liq.p, 1)
    liq.u = (liq.mdot/liq.rho)
    liq.p[1:] = (liq.p - liq.dx*vap.rho*np.gradient(liq.u, vap.x))[:-1]

# Check if the solver converged
if sol_liq.status == 0 and sol_gas.status == 0:
    print('Success: The solver converged.')
else:
    print('WARNING: The solver did not converge.')


plt.plot(vap.T - 273.15, vap.x * 1000)  # Convert x to mm
plt.plot(liq.T - 273.15, liq.x * 1000, 'C3')  # Convert x to mm
plt.plot(jafari_913Pa_40heat[:, 0], jafari_913Pa_40heat[:, 1],
         marker='>', lw=0, color='C2', label="Jafari et al. (913 Pa)")

plt.fill_between(np.linspace(5, 15), 0, L_liq*1000, alpha=0.5)
plt.xlabel('Temperature (deg. C)')
plt.ylabel('x [mm]')
plt.grid()
plt.legend()

# plt.xlim((-10, 20))
# plt.ylim((-10, 35))
plt.tight_layout()
plt.show()
