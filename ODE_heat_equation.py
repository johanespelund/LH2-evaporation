from scipy.optimize import fsolve
from force_flux_solver import solve_force_flux
from data_jafari import jafari_913Pa_40heat
import NET
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
psat = 913.15
Tgas = 273.15 + 6.1  # Temperature at x=0
Tliq = 273.15 + 5.7
p0 = thermo.calc_p_sat(Tliq, thermo.water)  # Initial pressure
TL = 273.15 + 40  # Temperature at x=L, assuming 60C for example
T_inf = Tliq + 2.5  # Superheat of bulk liquid, at L_liq
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
M = thermo.M_water

liq = Phase(L_liq, -0, N, water.LIQPH,
            lambda T, p: thermo.calc_cp(T, p, water.LIQPH, water),
            lambda T, p: thermo.calc_rho(T, p, water.LIQPH, water),
            lambda T, p: thermo.calc_kappa(T, p, water.LIQPH))
vap = Phase(0, L_gas, N, water.VAPPH,
            lambda T, p: thermo.calc_cp(T, p, water.VAPPH, water),
            lambda T, p: thermo.calc_rho(T, p, water.VAPPH, water),
            lambda T, p: thermo.calc_kappa(T, p, water.VAPPH))

liq.set_mdot(0)
vap.set_mdot(0)


# Boundary conditions


mdot = 1e-4
N_outer_iter = 10
for i in range(N_outer_iter):

    """
    Fix T_inf at bottom of liquid. The use initial guess for mdot and qliq
    to find Tliq (just below surface). 

    Use NET to calulate Tg (just above surface), using qliq and mdot.
    qvap is given by energy balance. Solve heat equation in gas phase.

    Use conductivity formulation of NET to update heat flux of gas and mass flux.
    Update qliq using current temperature jump, and start over.

    """
    dp = -1.8
    p_sat = thermo.calc_p_sat(Tliq, water)
    p0 = p_sat + 1.8

    vap.p = p0
    liq.p = p0

    def dTdx(x, y, phase: Phase):
        # T[0] is temperature, T[1] is its gradient
        T, dTdx = y
        d2Tdx2 = (phase.mdot * phase.calc_cp(T, phase.p) /
                phase.calc_kappa(T, phase.p)) * dTdx
        return np.vstack((dTdx, d2Tdx2))

    # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
    # Initial guess
    y0 = np.ones((2, N))*Tliq

    def bc_liq(ya, yb, phase: Phase):
        kappa_inlet = phase.calc_kappa(Tliq, p0)
        return np.array([ya[0] - T_inf, yb[1] + kappa_inlet*qliq])

    sol_liq = solve_bvp(lambda x, y: dTdx(x, y, liq),
                        lambda x, y: bc_liq(x, y, liq), liq.x, y0)
    liq.T = sol_liq.y[0]
    Tliq = liq.T[-1]
    


    # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")

    mdot, Tgas = solve_force_flux(Tliq, qgas, dp, water)
    # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")

    liq.set_mdot(mdot)
    vap.set_mdot(mdot)

    dH_vap = thermo.calc_dHvap(Tliq, Tgas, p0, water)
    qliq = qgas + mdot*dH_vap

    # print(f"Updated interface values of gas: {Tgas=}, {qgas=}")

    def bc_gas(ya, yb, phase: Phase):
        kappa_inlet = phase.calc_kappa(Tgas, phase.p)
        return np.array([ya[0] - Tgas, kappa_inlet*ya[1] + qgas])


    y0 = np.ones((2, N))*Tgas
    sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap),
                        lambda x, y: bc_gas(x, y, vap), vap.x, y0)
    vap.T = sol_gas.y[0]
    # vap.rho = vap.calc_rho(vap.T, vap.p)
    # vap.u = (vap.mdot/vap.rho)
    # vap.p[1:] = (vap.p - vap.dx*vap.rho*np.gradient(vap.u, vap.x))[:-1]
    # psat = thermo.calc_p_sat(Tliq, water)
    # print(f"{psat=}, {p0=}")


    # Check if the solver converged
    if sol_liq.status == 0 and sol_gas.status == 0:
        print('Success: The solver converged.')
    else:
        print('WARNING: The solver did not converge.')
    print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")


plt.plot(vap.T - 273.15, vap.x * 1000, label=f"p = {vap.p: .2f} Pa")  # Convert x to mm
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
