import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import NET
import thermo
from scipy.optimize import fsolve

R = thermo.R



def solve_force_flux(Tliq, qvap, delta_p, eos):
    p = thermo.calc_p_sat(Tliq, eos) + delta_p
    M = eos.compmoleweight(1)*1e-3
    # Initial guess
    J0 = 0
    Tv0 = Tliq

    def func(x):
        Tv, J = x # Vapor temp and molar flux
        p_sat = thermo.calc_p_sat(Tliq, eos)
        C_eq = thermo.calc_Ceq(Tv, eos)

        r_qq = NET.R_qq(C_eq, Tv, R, M)*1e2
        r_qmu = NET.R_qmu(C_eq, Tv, R, M)*1e3
        r_mumu = NET.R_mumu(C_eq, Tv, R, M)*1e5

        r1 = (1/Tv) - (1/Tliq) - NET.deltaT_inv(qvap, J, r_qq, r_qmu)
        r2 = -R*np.log(p/p_sat) - NET.p_by_psat(qvap, J, r_mumu, r_qmu)
        return np.array([r1, r2])

    res = fsolve(func, np.array([Tv0, J0]))
    Tv, J = res
    mdot = J/M
    return mdot, Tv

if __name__ == "__main__":
    plt.style.use(["science", "nature"])
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    a = axes.flatten()
    eos = thermo.water

    Tliq = 273 + 1.2
    print(thermo.calc_p_sat(Tliq, eos))
    qvap = -100.36
    delta_p = np.linspace(-2, 0)
    for qvap in [0, -0.1, -1, -10]:
        mdot, Tvap = np.zeros(delta_p.size), np.zeros(delta_p.size)
        for i in range(delta_p.size):
            dp = delta_p[i]
            
            mdot[i], Tvap[i] = solve_force_flux(Tliq, qvap, dp, eos)

        a[0].plot(delta_p, mdot,
                  label="{$q^\mathrm{g}$= " + f"{qvap: .2f} " + "W m$^{-2}$")
        a[0].set_ylabel("$\dot{m}$ [kg m$^{-2}$ s$^{-1}$]")
        a[1].plot(delta_p, Tvap-Tliq)
        a[1].set_ylabel("$T^\mathrm{g} - T^\ell$ [K]")
    for ax in a:
        ax.grid()
        ax.set_xlabel("$p - p_{sat}$ [Pa]")
    a[0].legend()
    fig.tight_layout()
    plt.show()

