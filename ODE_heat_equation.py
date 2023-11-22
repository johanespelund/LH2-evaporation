from scipy.optimize import fsolve
from force_flux_solver import solve_force_flux
import data_jafari as jafari
import data_badam as badam
import NET
import thermo
from KGT import mass_flux_HKS, sigma_condensation
from thermopack.cpa import cpa
from Phase import Phase
from CoolProp.CoolProp import PropsSI  # Import PropsSI from CoolProp
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature'])

# Parameters
psat = 913.15
Tgas = 273.15 + 6.1  # Temperature at x=0
Tliq = 273.15 + 5.6
p0 = thermo.calc_p_sat(Tliq, thermo.water)  # Initial pressure
TL = 273.15 + 40  # Temperature at x=L, assuming 60C for example
T_inf = 273.15 + jafari.T[7]  # Superheat of bulk liquid, at L_liq
L_gas = jafari.x[-1]*1e-3  # Length of the domain
L_liq = jafari.x[7]*1e-3

print(L_liq, L_gas)
print(T_inf)
N = 50
mdot = 0.39e-4  # Mass flow rate
qgas = -10.36  # Heat flux
qliq = 500.2  # Heat flux\

water = thermo.water
M = thermo.M_water

liq = Phase(L_liq, -0, N, water.LIQPH,
            lambda T, p: thermo.calc_cp(T, p, water.LIQPH, water),
            lambda T, p: thermo.calc_rho(T, p, water.LIQPH, water),
            lambda T, p, x: thermo.calc_kappa(T, p, water.LIQPH, x))
vap = Phase(0, L_gas, N, water.VAPPH,
            lambda T, p: thermo.calc_cp(T, p, water.VAPPH, water),
            lambda T, p: thermo.calc_rho(T, p, water.VAPPH, water),
            lambda T, p, x: thermo.calc_kappa(T, p, water.VAPPH, x))

liq.set_mdot(mdot)
vap.set_mdot(mdot)

# Boundary conditions

print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
N_outer_iter = 10
for label, rauter in zip(["KTG (fitted scale)", "Rauter et al."], [False, True]):
    for i in range(N_outer_iter):

        dp = jafari.dp
        p_sat = thermo.calc_p_sat(Tliq, water)
        p0 = p_sat + 1.8

        vap.p = p0
        liq.p = p0

        def dTdx(x, y, phase: Phase):
            # T[0] is temperature, T[1] is its gradient
            T, dTdx = y
            kappa = phase.calc_kappa(T, phase.p, x)#*(1 + 100*((x > -2.5e-3) & (x<0)))
            dkdx = np.gradient(kappa, x)
            # d2Tdx2 = dTdx*(phase.mdot * phase.calc_cp(T, phase.p) /
            #          (kappa))
            d2Tdx2 = ((phase.mdot * phase.calc_cp(T, phase.p) - 1*dkdx)*
                    (dTdx/kappa))
            # d2Tdx2 = ((phase.mdot * phase.calc_cp(T, phase.p) - 0*dkdx)*
            #          (1/kappa))
            # print(kappa)
            # input()
            return np.vstack((dTdx, d2Tdx2))

        # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
        # Initial guess
        y0 = np.ones((2, N))*Tliq

        def bc_liq(ya, yb, phase: Phase):
            kappa_inlet = phase.calc_kappa(Tliq, p0, 0)
            return np.array([ya[0] - T_inf, yb[1] + kappa_inlet*qliq])

        sol_liq = solve_bvp(lambda x, y: dTdx(x, y, liq),
                            lambda x, y: bc_liq(x, y, liq), liq.x, y0)
        liq.T = sol_liq.y[0]
        # liq.x = sol_liq.x
        Tliq = liq.T[-1]
        


        # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
        f1=71.44
        f2=304308
        print(Tliq, qgas, dp, water, f1, f2)
        mdot, Tgas = solve_force_flux(Tliq, qgas, dp, water, f1=f1, f2=f2, RAUTER=rauter)
        print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
        # exit()

        liq.set_mdot(mdot)
        vap.set_mdot(mdot)

        dH_vap = thermo.calc_dHvap(Tliq, Tgas, p0, water)
        qliq = qgas + mdot*dH_vap

        # print(f"Updated interface values of gas: {Tgas=}, {qgas=}")

        def bc_gas(ya, yb, phase: Phase):
            kappa_inlet = phase.calc_kappa(Tgas, phase.p, 0)
            return np.array([ya[0] - Tgas, kappa_inlet*ya[1] + qgas])


        y0 = np.ones((2, N))*Tgas
        sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap),
                            lambda x, y: bc_gas(x, y, vap), vap.x, y0)
        vap.T = sol_gas.y[0]
        # vap.x = sol_gas.x
        # vap.rho = vap.calc_rho(vap.T, vap.p)
        # vap.u = (vap.mdot/vap.rho)
        # vap.p[1:] = (vap.p - vap.dx*vap.rho*np.gradient(vap.u, vap.x))[:-1]
        # psat = thermo.calc_p_sat(Tliq, water)
        # print(f"{psat=}, {p0=}")


        # Check if the solver converged
        if sol_liq.status != 0 or sol_gas.status != 0:
            print('WARNING: The solver did not converge.')

        print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")

    liq.x = sol_liq.x
    vap.x = sol_gas.x

    plt.plot(vap.T - 273.15, vap.x * 1000, f'C{0+rauter}', label=label)  # Convert x to mm
    plt.plot(liq.T - 273.15, liq.x * 1000, f'C{0+rauter}')  # Convert x to mm

ax = plt.gca()

axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
             origin="lower")

# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.plot(jafari.T, jafari.x,
         marker='>', lw=0, color='C2', label="Jafari et al.")
# plt.plot(badam.T, badam.x, 'C2o', label="Badam et al. (2007)")
plt.fill_between(np.linspace(-2, 25), 0, L_liq*1000, alpha=0.5)
plt.xlabel(r'T [$^\circ$C]')
plt.ylabel('x [mm]')
plt.grid()
plt.legend()

# plt.xlim((-10, 20))
# plt.ylim((-10, 35))
plt.tight_layout()
plt.show()
