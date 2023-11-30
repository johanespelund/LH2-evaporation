import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import NET, KGT
import thermo
import kazemi_thermo as k_thermo
from scipy.optimize import fsolve

R = thermo.R

def solve_force_flux(Tliq, qvap, dp, eos, DOF, f1=1, f2=1, method="KTG"):
    M = eos.compmoleweight(1)*1e-3
    p_sat = k_thermo.psat_liquid(Tliq)
    p = p_sat + dp
    C_eq = k_thermo.Ceq(Tliq)
    
    # Uncomment for other components than water!
    # print(f"{Tliq=}")
    # M = eos.compmoleweight(1)*1e-3
    # p_sat = thermo.calc_p_sat(Tliq, eos)
    # p = p_sat + dp
    # C_eq = thermo.calc_Ceq(Tliq, eos)
        
    def func(x):
        J, Tg = x

        if method == "RAUTER":
            r_qq = -1.5636e-8*Tliq + 4.6189e-6
            r_qmu = -0.0026*Tliq + 0.7415
            r_mumu = -2.7399e-3*Tliq + 8.0423e5
        
        elif method == "KTG":
            r_qq = KGT.R_qq(C_eq, Tliq, R, M)*f1
            r_qmu = KGT.R_qmu(C_eq, Tliq, R, M)*f1
            r_mumu = KGT.R_mumu(C_eq, Tliq, R, M, Tg, DOF, sigma=False)*f2
    
        force1, force2 = NET.forces(qvap, J, r_qq, r_mumu, r_qmu)
        r1 = (1/Tg - 1/Tliq) - force1
        r2 = -R*np.log(p/p_sat) - force2
        # The resiudals r1 and r2 should be zero when the solution is found
        return np.array([r1, r2])
    
    y0 = np.array([1e-6, Tliq]) # Initial guess for mass flux and vapor interface temperature
    J, Tg = fsolve(func, y0)    # [mol/(m^2 s)], [K]
    
    mdot = J/M
    return mdot, Tg # [kg/(m^2 s)], [K]


if __name__ == "__main__":
    plt.style.use(["science", "nature"])
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    a = axes.flatten()
    
    import data.jafari as data
    
    ### water
    eos = thermo.water
    DOF = thermo.DOF_water
    Tliq = 273.15 + data.T_liq[0]
    dp = data.dp
    qvap = data.q_gas
    
    target_mdot = data.mdot
    target_Tv = 273.15 + data.T_gas[0]
    
    ### N2
    # eos = thermo.N2
    # DOF = thermo.DOF_N2
    # Tliq = 77.6
    # dp = 2
    # qvap = -10
    
    # target_mdot = 13.1e-3
    # target_Tv = Tliq + 3.2
    
    # Solve func(f1, f2) == 0 to find scaling factors f1 and f2 
    # that makes KTG fit with experiments:
    if eos == thermo.water:
        def func2(x):
            f1, f2 = x
            mdot, Tv = solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
            return np.array([target_mdot - mdot, target_Tv  - Tv])
        f1, f2 = fsolve(func2, [1, 1])
        
        print(f1, f2)
        
        # f1, f2 = 70, 3e5 # Best values for Jafari et al.
        # f1, f2 = 1e3, 1e3
        
        mdot1, Tvap1= solve_force_flux(Tliq, qvap, dp, eos, DOF)
        print(mdot1, Tvap1-Tliq)
        mdot2, Tvap2= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
        print(mdot2, Tvap2-Tliq)
        mdot3, Tvap3= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1=1, f2=1, method="RAUTER")
        print(mdot3, Tvap3-Tliq)
        mdot4, Tvap4 = data.mdot, 273.15 + data.T_gas[0]    # Jafari
        
        labels = ["Jafari et al.", "KTG", "KTG (scaled)", "Rauter et al."]
        colors = ["C3", "C0", "C0", "C0"]
            
        a[0].bar([1, 2, 3, 4], [mdot4, mdot1, mdot2, mdot3], color=colors, zorder=3)
        a[1].bar([1, 2, 3, 4], [Tvap4-Tliq, Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq],  color=colors, zorder=3)
        
        for ax in a:
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(axis='y', zorder=0)
    
    else:
        
        # def func2(x):
        #     f1, f2 = x
        #     mdot, Tv = solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
        #     return np.array([target_mdot - mdot, target_Tv  - Tv])
        # f1, f2 = fsolve(func2, [1, 1])
        
        f1, f2 = 1e3, 1e9
        print(f1, f2)
        mdot1, Tvap1= solve_force_flux(Tliq, qvap, dp, eos, DOF)
        print(mdot1, Tvap1-Tliq)
        mdot2, Tvap2= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
        print(mdot2, Tvap2-Tliq)
        mdot3, Tvap3= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1=1, f2=1, method="RAUTER")
        print(mdot3, Tvap3-Tliq)
        mdot4, Tvap4 = target_mdot, target_Tv    # Jafari
        
        labels = ["KTG", "KTG (scaled)", "Rauter et al."]
        colors = ["C1", "C0", "C0", "C0"]
            
        a[0].bar([1, 2, 3, 4], [mdot4, mdot1, mdot2, mdot3], color=colors, zorder=3)
        a[1].bar([1, 2, 3, 4], [Tvap4-Tliq, Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq],  color=colors, zorder=3)
        
        # for ax in a:
        #     ax.set_xticks([1, 2, 3, 4])
        #     ax.set_xticklabels(labels, rotation=45, ha='right')
        #     ax.grid(axis='y', zorder=0)
        
        
        
    a[0].set_yscale("log")
    a[0].set_ylabel(r"$\dot{m}$ [kg/s]")
    a[1].set_ylabel("$T^g - T^\ell$ [K]")
    fig.tight_layout()
    plt.show()


