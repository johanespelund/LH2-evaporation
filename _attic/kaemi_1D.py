from force_flux_solver import solve_force_flux
import data.badam as badam
import kazemi_thermo as thermo
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
p0 = thermo.psat_liquid(Tliq)  # Initial pressure
TL = 273.15 + 40  # Temperature at x=L, assuming 60C for example
T_inf = 273.15 + 4.5 
L_gas = 4e-3  # Length of the domain
L_liq = -2e-3
N = 50
Di = 11e-3
Do = 12e-3
mdot0 = 3.78e-4  # Mass flow rate
# mdot = mdot_HKS
qgas = -15.56  # Heat flux
qliq = 500  # Heat flux\

Lqq = 0.236e-4
Lww = 24.77e-8
Lqw = -0.111

# mdot0 = mass_flux_NET(T0, Tliq, p0, psat, 8.314, M, Lww, Lqw)
# qgas = heat_flux_gas_NET(T0, Tliq, p0, psat, 8.314, M, Lqq, Lqw)

# print(mdot0, qgas)


# ODE system (in first-order form)


liq = Phase(L_liq, -0, N, 0,
            lambda T, p: thermo.cp_liquid(T),
            lambda T, p: thermo.rho_liquid(T),
            lambda T, p, x: thermo.kappa_liquid(T))
vap = Phase(0, L_gas, N, 1,
            lambda T, p: thermo.cp_vapor(T),
            lambda T, p: thermo.rho_vapor(T, p),
            lambda T, p, x: thermo.kappa_vapor(T))



# Boundary conditions


mdot = 1.1337e-4
liq.set_mdot(mdot)
vap.set_mdot(mdot)
print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
N_outer_iter = 25
for i in range(N_outer_iter):

    # dp = 0
    # p_sat = thermo.calc_p_sat(Tliq)
    # p0 = p_sat + 1.8

    vap.p = p0
    liq.p = p0

    def dTdx(x, y, phase: Phase):
        # T[0] is temperature, T[1] is its gradient
        T, dTdx = y
        kappa = phase.calc_kappa(T, phase.p, x)#*(1 + 100*((x > -2.5e-3) & (x<0)))
        print(kappa[0], phase.calc_cp(T, phase.p)[0])
        dkdx = np.gradient(kappa, x)
        dx = np.gradient(x)
        # d2Tdx2 = dTdx*(phase.mdot * phase.calc_cp(T, phase.p) /
        #          (kappa))
        S = 0#2*np.pi*dx*thermo.k_solid("borosilicate_glass")*(298 - T)/np.log(Do/Di)
        # print(S[0])
        d2Tdx2 = (((phase.mdot * phase.calc_cp(T, phase.p))*dTdx - dkdx - S)/kappa)
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

    # mdot, Tgas = solve_force_flux(Tliq, qgas, dp, water)
    # print(f"{mdot=}, {qliq=}, {qgas=}, {Tliq=}, {Tgas=}")
    Tgas = Tliq
    # liq.set_mdot(mdot)
    # vap.set_mdot(mdot)

    dH_vap = 45000/18e-3 #thermo.calc_dHvap(Tliq)
    qliq = qgas + vap.mdot*dH_vap

    # print(f"Updated interface values of gas: {Tgas=}, {qgas=}")

    def bc_gas(ya, yb, phase: Phase):
        kappa_inlet = phase.calc_kappa(Tgas, phase.p, 0)
        return np.array([ya[0] - Tgas, kappa_inlet*ya[1] + qgas])


    y0 = np.ones((2, N))*Tgas
    sol_gas = solve_bvp(lambda x, y: dTdx(x, y, vap),
                        lambda x, y: bc_gas(x, y, vap), vap.x, y0)
    vap.T = sol_gas.y[0]
    vap.x = sol_gas.x
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

plt.plot(vap.T - 273.15, vap.x * 1000, label=f"NET")  # Convert x to mm
plt.plot(liq.T - 273.15, liq.x * 1000, 'C3')  # Convert x to mm
# plt.plot(jafari_913Pa_40heat[:, 0], jafari_913Pa_40heat[:, 1],
#          marker='>', lw=0, color='C2', label="Jafari et al. (913 Pa)")
# plt.plot(badam.T, badam.x, 'C2o', label="Badam et al. (2007)")
# plt.fill_between(np.linspace(-2, 25), 0, L_liq*1000, alpha=0.5)
plt.xlabel(r'T [$^\circ$C]')
plt.ylabel('x [mm]')
plt.grid()
plt.legend()

# plt.xlim((-10, 20))
# plt.ylim((-10, 35))
plt.tight_layout()
plt.show()
