import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import NET
import thermo
from scipy.optimize import fsolve

R = thermo.R

def solve_force_flux(Tliq, qvap, delta_p, eos, f1=1, f2=1, RAUTER=False):
    p = thermo.calc_p_sat(Tliq, eos) + delta_p
    M = eos.compmoleweight(1)*1e-3
    # Initial guess
    p_sat = thermo.calc_p_sat(Tliq, eos)
    C_eq = thermo.calc_Ceq(Tliq, eos)

    if not RAUTER:
        r_qq = NET.R_qq(C_eq, Tliq, R, M)*f1
        # r_qmu = -0.0*r_qq #NET.R_qmu(C_eq, Tliq, R, M)*FACTOR
        r_qmu = NET.R_qmu(C_eq, Tliq, R, M)*f1
        r_mumu = NET.R_mumu(C_eq, Tliq, R, M)*f2
    
    else:
        r_qq = -1.5636e-8*Tliq + 4.6189e-6
        # r_qmu = -0.0*r_qq #NET.R_qmu(C_eq, Tliq, R, M)*FACTOR
        r_qmu = -0.0026*Tliq + 0.7415
        r_mumu = -2.7399e-3 + 8.0423e5
    
    J = (-R*np.log(p/p_sat) - r_qmu*qvap)/r_mumu
    Tv = (r_qq*qvap + r_qmu*J + 1/Tliq)**-1
    
    mdot = J/M
    return mdot, Tv


if __name__ == "__main__":
    plt.style.use(["science", "nature"])
    # fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    # a = axes.flatten()
    eos = thermo.water

    Tliq = 273 + 5.2
    print(thermo.calc_p_sat(Tliq, eos))
    qvap = -10.36
    delta_p = -1.8 #np.linspace(-2, 0)
    # for qvap in [0, qvap/100, qvap/10, qvap]:
    #     mdot, Tvap = np.zeros(delta_p.size), np.zeros(delta_p.size)
    #     for i in range(delta_p.size):
            # dp = delta_p[i]
    dp = delta_p
    
    
    def func2(x):
        f1, f2 = x
        mdot, Tv = solve_force_flux(Tliq, qvap, dp, eos, f1, f2)
        return np.array([0.39e-4 - mdot, 273.15 + 5.6 - Tv])
    
    f1, f2 = fsolve(func2, [1, 1])
    f1, f2 = 1e3, 1e3
    # print(f1, f2)
    mdot1, Tvap1= solve_force_flux(Tliq, qvap, dp, eos)
    print(mdot1, Tvap1)
    # print(Tliq, qvap, dp, eos, f1, f2)
    mdot2, Tvap2= solve_force_flux(Tliq, qvap, dp, eos, f1, f2)
    print(mdot2, Tvap2)
    mdot3, Tvap3= solve_force_flux(Tliq, qvap, dp, eos, f1, f2, RAUTER=True)
    print(mdot2, Tvap2)
    mdot4, Tvap4 = 2.45e-4, 273.15 + 5.6
    # print(f"{mdot=}, {Tvap-Tliq=}")
    
    labels = ["Jafari et al.", "KTG", "KTG (scaled 1e3)", "Rauter et al."]
    plt.bar([1,2,3,4], [mdot4, mdot1, mdot2, mdot3])
    # plt.bar([1, 2, 3, 4], [Tvap4-Tliq, Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq])
    plt.gca().set_xticks([1, 2, 3, 4])
    plt.gca().set_xticklabels(labels, rotation=45, ha='right')
    plt.yscale("log")
    plt.ylabel(r"$\dot{m}$ [kg/s]")
    # plt.ylabel("$T^g - T^\ell$ [K]")
    plt.tight_layout()
    plt.show()

        # a[0].plot(delta_p, mdot,
        #           label="{$q^\mathrm{g}$= " + f"{qvap: .2f} " + "W m$^{-2}$")
        # a[0].set_ylabel("$\dot{m}$ [kg m$^{-2}$ s$^{-1}$]")
        # a[1].plot(delta_p, Tvap-Tliq)
        # a[1].set_ylabel("$T^\mathrm{g} - T^\ell$ [K]")
    # for ax in a:
    #     ax.grid()
    #     ax.set_xlabel("$p - p_{sat}$ [Pa]")
    # a[0].legend()
    # fig.tight_layout()
    # plt.show()

