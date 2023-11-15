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
T_inf = 273.15 + 1  # Superheat of bulk liquid, at L_liq
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
N_outer_iter = 1
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

    vap.p[:] = p0
    liq.p[:] = p0

    def dTdx(x, y, phase: Phase, mdot):
        # T[0] is temperature, T[1] is its gradient
        T, dTdx = y
        print(T, dTdx, phase.p)
        print()
        dudx = (mdot * phase.calc_cp(T, phase.p) /
                phase.calc_kappa(T, phase.p)) * dTdx
        return np.vstack((dTdx, dudx))

    print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
    # Initial guess
    y0 = np.ones((2, N))*Tliq

    # qliq = qvap + mdot0*dH_vap

    def bc_liq(ya, yb, phase: Phase):
        kappa_inlet = phase.calc_kappa(Tliq, p0)
        return np.array([ya[0] - T_inf, yb[0] + kappa_inlet*qliq])

    sol_liq = solve_bvp(lambda x, y: dTdx(x, y, liq, mdot),
                        lambda x, y: bc_liq(x, y, liq), liq.x, y0)
    liq.T = sol_liq.y[0]
    Tliq = liq.T[-1]
    print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
    # liq.rho = liq.calc_rho(liq.T, liq.p)
    # liq.u = (liq.mdot/liq.rho)
    # liq.p[1:] = (liq.p - liq.dx*vap.rho*np.gradient(liq.u, vap.x))[:-1]

    mdot0, Tgas = solve_force_flux(Tliq, qgas, dp, water)

    # r_qq = 0.5e-6
    # r_qmu = -1
    # r_mumu = 2e6

    # R = 8.314

    # def func(x):
    #     _Tg, _mdot  = x
    #     vg, = water.specific_volume(_Tg, thermo.calc_p_sat(Tliq, water), [1], water.VAPPH) # Molar volume of gas phase (NB: Notice the comma)
    #     C_eq = 1/vg     #  mol/m^3
    #     print(f"{C_eq=}")
    #     print(_Tg, _mdot, C_eq)
    #     r_qq = NET.R_qq(C_eq, Tliq, R, M)
    #     r_qmu = NET.R_qmu(C_eq, Tliq, R, M)
    #     r_mumu = NET.R_mumu(C_eq, Tliq, R, M)
    #     r1 = (1/_Tg) - (1/Tliq) - NET.deltaT_inv(qliq, _mdot/M, r_qq, r_qmu)
    #     r2 = np.log(p0/thermo.calc_p_sat(_Tg, water)) - NET.p_by_psat(qliq, _mdot/M, r_qmu, r_mumu, 8.314)
    #     return np.array([r1, r2])

    # res = fsolve(func, np.array([Tgas, mdot]))
    # print(res)
    # exit()

    # Tgas = (1/Tliq + NET.deltaT_inv(qliq, mdot, 2e-10, -5e-12))**-1
    dH_vap = thermo.calc_dHvap(Tliq, Tgas, p0, water)
    qliq = qgas + mdot*dH_vap

    print(f"Updated interface values of gas: {Tgas=}, {qgas=}")

    def bc_gas(ya, yb, phase: Phase):
        kappa_inlet = phase.calc_kappa(Tgas, p0)
        return np.array([ya[0] - Tgas, -kappa_inlet*ya[1] - qgas])

    # Solve BVP

    sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap, mdot),
                        lambda x, y: bc_gas(x, y, vap), vap.x, y0)
    vap.T = sol_gas.y[0]
    # vap.rho = vap.calc_rho(vap.T, vap.p)
    # vap.u = (vap.mdot/vap.rho)
    # vap.p[1:] = (vap.p - vap.dx*vap.rho*np.gradient(vap.u, vap.x))[:-1]
    # psat = thermo.calc_p_sat(Tliq, water)
    # print(f"{psat=}, {p0=}")

    # mdot = NET.mass_flux_NET(Tgas, Tliq, p0, psat, 8.314, thermo.M, Lww, Lqw)
    # qgas = NET.heat_flux_gas_NET(Tgas, Tliq, p0, psat, 8.314, thermo.M, Lqq, Lqw)
    # qliq = qgas + mdot*dH_vap

    print(f"Updated interface values again: {mdot=}, {qliq=}, {qgas=}")

    # Check if the solver converged
    # if sol_liq.status == 0 and sol_gas.status == 0:
    #     print('Success: The solver converged.')
    # else:
    #     print('WARNING: The solver did not converge.')


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
