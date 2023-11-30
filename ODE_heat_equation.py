from force_flux_solver import solve_force_flux
import thermo
import kazemi_thermo as k_thermo
from Phase import Phase
import numpy as np
from scipy.integrate import solve_bvp

import data.jafari as data # Used for experimental reference

# Visualization
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])

# Parameters
# Tgas = 273.15 + data.T_gas[0]
# Tliq = 273.15 + data.T_liq[0]
# dp = -1.8
# T_inf = 273.15 + data.T_liq[0]    # Superheat of bulk liquid, at L_liq
# L_gas = data.x_gas[-1]*1e-3       # Length of the domain
# L_liq = data.x_liq[-1]*1e-3        # Liquid temperature just below interface

N = 50
# mdot = 0.39e-4  # Mass flow rate (guess)
# qgas = -10.36  # Heat flux (from experiment)
# qliq = 500.2  # Heat flux (guess)

# eos = thermo.H2
# M = thermo.M_H2
# DOF = thermo.DOF_H2


Tgas = 273.15 + data.T_gas[0]
Tliq = 273.15 + data.T_liq[0]
p0 = 545
T_inf = 273.15 + data.T_liq[-1]   # Superheat of bulk liquid, at L_liq
L_gas = data.x_gas[-1]*1e-3       # Length of the domain
L_liq = data.x_liq[-1]*1e-3        # Liquid temperature just below interface

mdot = data.mdot  # Mass flow rate (guess)
qgas = -10.36  # Heat flux (from experiment)
dTdx_gas = 1e3*np.gradient(data.T_gas, data.x_gas)[0]
qgas = -k_thermo.kappa_vapor(Tgas)*dTdx_gas
dTdx_liq = 1e3*np.gradient(data.T_liq, data.x_liq)[0]
qliq = -k_thermo.kappa_liquid(Tliq)*dTdx_liq

eos = thermo.water
M = thermo.M_water
DOF = thermo.DOF_water

dp = -1.8

# liq = Phase(L_liq, -0, N, eos.LIQPH,
#             lambda T, p: thermo.calc_cp(T, p, eos.LIQPH, eos),
#             lambda T, p: thermo.calc_rho(T, p, eos.LIQPH, eos),
#             lambda T, p, x: thermo.calc_kappa(T, p, eos.LIQPH, x))
# vap = Phase(0, L_gas, N, eos.VAPPH,
#             lambda T, p: thermo.calc_cp(T, p, eos.VAPPH, eos),
#             lambda T, p: thermo.calc_rho(T, p, eos.VAPPH, eos),
#             lambda T, p, x: thermo.calc_kappa(T, p, eos.VAPPH, x))

liq = Phase(L_liq, -0, N, eos.LIQPH,
            lambda T, p: k_thermo.cp_liquid(T),
            lambda T, p: k_thermo.rho_liquid(T),
            lambda T, p, x: k_thermo.kappa_liquid(T))
vap = Phase(0, L_gas, N, eos.VAPPH,
            lambda T, p: k_thermo.cp_vapor(T),
            lambda T, p: k_thermo.rho_vapor(T),
            lambda T, p, x: k_thermo.kappa_vapor(T))

liq.set_mdot(mdot)
vap.set_mdot(mdot)

# Boundary conditions

fig, ax = plt.subplots(1,1)

N_outer_iter = 10
i = 0
for label, method in zip(["KTG (fitted scale)", "Rauter et al."], ["KTG", "RAUTER"]):
    mdot0 = 0
    i = 0
    while np.abs(mdot0 - liq.mdot) > 1e-12:
        i += 1

        vap.p = data.p
        liq.p = data.p

        def dTdx(x, y, phase: Phase):
            # T[0] is temperature, T[1] is its gradient
            T, dTdx = y
            # print(T)
            kappa = phase.calc_kappa(T, phase.p, x)
            dkdx = np.gradient(kappa, x)
            d2Tdx2 = ((phase.mdot * phase.calc_cp(T, phase.p) - 1*dkdx)*
                    (dTdx/kappa))
            return np.vstack((dTdx, d2Tdx2))

        # Initial guess
        y0 = np.ones((2, N))*Tliq

        def bc_liq(ya, yb, phase: Phase):
            kappa_inlet = phase.calc_kappa(Tliq, p0, 0)
            return np.array([yb[0] - Tliq, yb[1] + kappa_inlet*qliq])

        sol_liq = solve_bvp(lambda x, y: dTdx(x, y, liq),
                            lambda x, y: bc_liq(x, y, liq), liq.x, y0)
        liq.T = sol_liq.y[0]
        Tliq = liq.T[-1]
        
        f1=71.44
        f2=304308
        # input()
        mdot, Tgas = solve_force_flux(Tliq, qgas, data.p, eos, DOF, f1=f1, f2=f2, method=method)
        # mdot, Tgas = data.mdot, Tgas 

        mdot0 = liq.mdot
        liq.set_mdot(mdot)
        vap.set_mdot(mdot)

        u_gas = mdot/k_thermo.rho_vapor(Tgas, vap.p)
        u_liq = mdot/k_thermo.rho_liquid(Tgas)
        print(u_gas, u_liq)
        dH_vap = k_thermo.dH_vap(Tliq)
        deltaP = mdot*(u_gas - u_liq)
        print(deltaP)
        qliq = qgas + mdot*dH_vap #+ 0.5*(u_liq**2 - u_gas**2)
        
        print(k_thermo.psat_liquid(Tliq), data.p)

        def bc_gas(ya, yb, phase: Phase):
            kappa_inlet = phase.calc_kappa(Tgas, phase.p, 0)
            return np.array([ya[0] - Tgas, kappa_inlet*ya[1] + qgas])


        y0 = np.ones((2, N))*Tgas
        sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap),
                            lambda x, y: bc_gas(x, y, vap), vap.x, y0)
        vap.T = sol_gas.y[0]


        # Check if the solver converged
        if sol_liq.status != 0 or sol_gas.status != 0:
            print('WARNING: The solver did not converge.')

        print(f"Iteration {i} ({method}): {mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")

    liq.x = sol_liq.x
    vap.x = sol_gas.x
    rauter = method == "RAUTER"
    ax.plot(vap.T - 273.15, vap.x * 1000, f'C{0+3*rauter}', label=label)  # Convert x to mm
    ax.plot(liq.T - 273.15, liq.x * 1000, f'C{0+3*rauter}')  # Convert x to mm
    
    # if not rauter:
    #     axins = zoomed_inset_axes(ax, 4, loc=4) # zoom = 6
    #     axins.plot(data.T, data.x,
    #         marker='>', lw=0, color='C2')
    # axins.plot(vap.T - 273.15, vap.x * 1000, f'C{0+3*rauter}')
    # axins.plot(liq.T - 273.15, liq.x * 1000, f'C{0+3*rauter}')
# axins.fill_between(np.linspace(5, 17), 0, L_liq*1000, alpha=0.4)
# axins.grid()


ax.plot(data.T_gas, data.x_gas,
         marker='>', lw=0, color='C2', label=data.label)
ax.plot(data.T_liq, data.x_liq,
         marker='>', lw=0, color='C2')
ax.plot(data.T_liq[0], data.x_liq[0], 'ko')
# ax.fill_between(np.linspace(5, 17), 0, L_liq*1000, alpha=0.4)
ax.set_xlabel(r'T [$^\circ$C]')
ax.set_ylabel('x [mm]')
ax.grid()
ax.legend()

# plt.xlim((-10, 20))
# plt.ylim((-10, 35))
plt.tight_layout()

# sub region of the original image
# x1, x2, y1, y2 = 5.5, 7, -1, 1
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=False)
# plt.yticks(visible=False)
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.3")

plt.show()
