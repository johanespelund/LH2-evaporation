import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import NET
import thermo
from scipy.optimize import fsolve

plt.style.use(["science", "nature"])
fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
a = axes.flatten()


def deltaT_inv(q_g, J, rqq_sg, rqmu_sg):
    return rqq_sg*q_g + rqmu_sg*J


def p_by_psat(q_g, J, rmumu_sg, rqmu_sg):
    return rqmu_sg*q_g + rmumu_sg*J


eos = thermo.water
M = thermo.M_water
R = 8.314


Tliq = 273 + 1.2
qvap = -100.36
delta_p = np.linspace(-2, 0)
for qvap in [0, -1, -10, -100]:
    mdot, Tvap = np.zeros(delta_p.size), np.zeros(delta_p.size)
    for i in range(delta_p.size):
        dp = delta_p[i]
        p = thermo.calc_p_sat(Tliq, eos) + dp

        J0 = 0
        Tv0 = Tliq

        def func(x):
            _Tv, J = x
            p_sat = thermo.calc_p_sat(Tliq, eos)
            C_eq = thermo.calc_Ceq(_Tv, eos)

            r_qq = NET.R_qq(C_eq, _Tv, R, M)
            r_qmu = NET.R_qmu(C_eq, _Tv, R, M)
            r_mumu = NET.R_mumu(C_eq, _Tv, R, M)*1e3

            r1 = (1/_Tv) - (1/Tliq) - deltaT_inv(qvap, J, r_qq, r_qmu)
            r2 = -R*np.log(p/p_sat) - \
                p_by_psat(qvap, J, r_mumu, r_qmu)
            return np.array([r1, r2])

        res = fsolve(func, np.array([Tv0, J0]))
        Tv, J = res
        mdot[i] = J/M
        Tvap[i] = Tv

    a[0].plot(delta_p, mdot, label="{$q^\mathrm{g}$= " + f"{qvap: .2f} " + "W m$^{-2}$")
    a[0].set_ylabel("$\dot{m}$ [kg m$^{-2}$ s$^{-1}$]")
    a[1].plot(delta_p, Tvap-Tliq)
    a[1].set_ylabel("$T^\mathrm{g} - T^\ell$ [K]")
for ax in a:
    ax.grid()
    ax.set_xlabel("$p - p_{sat}$ [Pa]")
a[0].legend()
fig.tight_layout()
plt.show()

# print(Tliq, Tvap)
# print(f"{mdot=}")
# print(func([Tliq, J]))
