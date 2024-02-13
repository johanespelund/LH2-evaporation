
import numpy as np
import NET, KGT
import thermo
import kazemi_thermo as k_thermo
from scipy.optimize import fsolve

R = thermo.R

def solve_force_flux(Tliq, qvap, dp, eos, DOF, f1=1, f2=1, method="KTG"):
    M = eos.compmoleweight(1)*1e-3
    p_sat = thermo.calc_p_sat(Tliq, eos)
    p = p_sat + dp
    C_eq = thermo.calc_Ceq(Tliq, eos)
        
    def func(x):
        J, Tg = x
        Tc_H2 = 32.938
        Tc_water = NET.Ts_water
        Tc = Tc_water
        if method == "RAUTER":
            r_qq = NET.r_qq(Tliq, Tc) #-1.5636e-8*Tliq + 4.6189e-6
            r_qmu = NET.r_qmu(Tliq, Tc)#*M #-0.0026*Tliq + 0.7415
            r_mumu = NET.r_mumu(Tliq, Tc)#*M**2 #-2.7399e-3*Tliq + 8.0423e5  
            force1, force2 = NET.forces(qvap, J*M, r_qq, r_mumu, r_qmu)
            r1 = (1/Tg - 1/Tliq) - force1
            r2 = -(R/M)*np.log(p/p_sat) - force2
        
        elif method == "KTG":
            r_qq = KGT.R_qq(C_eq, Tliq, R, M)*f1
            r_qmu = KGT.R_qmu(C_eq, Tliq, R, M)*f2#0.5*(f1+f2)
            r_mumu = KGT.R_mumu(C_eq, Tliq, R, M, Tg, DOF, sigma=1)*f2  

            
            force1, force2 = NET.forces(qvap, J, r_qq, r_mumu, r_qmu)
                
            r1 = (1/Tg - 1/Tliq) - force1
            r2 = -R*np.log(p/p_sat) - force2
            
        constratins = r_qq > 0 and r_mumu > 0  and r_qq*r_mumu - r_qmu**2 > 0
        if not constratins:
            print("Coefficients violate second law!")
            print(f"{Tliq=}, {qvap=}, {r_qq > 0=}, {r_mumu > 0=}  and {r_qq*r_mumu - r_qmu**2 > 0=}")
            
        # The resiudals r1 and r2 should be zero when the solution is found
        return np.array([r1, r2])

    

    
    y0 = np.array([1e-6, 1.01*Tliq]) # Initial guess for mass flux and vapor interface temperature
    J, Tg = fsolve(func, y0)    # [mol/(m^2 s)], [K]
    
    
    mdot = J*M # convert to kg based mass flux
    return mdot, Tg # [kg/(m^2 s)], [K]


if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('pgf')

    import matplotlib.pyplot as plt
    import scienceplots
    import plot_tools
    plt.style.use(["science", "nature"])
#     plt.rcParams.update({
#     "font.family": "sans-serif",  # use serif/main font for text elements
#     "text.usetex": True,     # use inline math for ticks,
#     'font.size' : 11,
#     "font.size": 11,  # Match LaTeX document's font size
#     "axes.labelsize": 11,
#     "legend.fontsize": 11,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "pgf.rcfonts": False     # don't setup fonts from rc parameters
#     })
    
#     plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": "\n".join([
#          r"\usepackage[utf8x]{inputenc}",
#          r"\usepackage[T1]{fontenc}",
#          r"\usepackage[default,scale=0.95]{carlito}"
#     ]),
# })
    

    fig, axes = plt.subplots(1, 2, figsize=plot_tools.set_size(416.83289, subplots=(1,2)))#, sharey=True)
    a = axes.flatten()
    mdot_label = r"$m''$ [kg m$^{-2}$s$^{-1}$]"
    
    import data.jafari as data
    
    ### water
    # eos = thermo.water
    
    ### N2
    # eos = thermo.N2

    
    ### H2
    eos = thermo.H2
    
    
    # Solve func(f1, f2) == 0 to find scaling factors f1 and f2 
    # that makes KTG fit with experiments:
    if eos == thermo.water:
        
        DOF = thermo.DOF_water
        Tliq = 273.15 + data.T_liq[0]
        p_by_psat = data.p_by_psat
        qvap = data.q_gas
        # print(p_by_psat, qvap)
        
        target_mdot = data.mdot
        target_Tv = 273.15 + data.T_gas[0]
        
        def func2(x):
            f1, f2 = x
            mdot, Tv = solve_force_flux(Tliq, qvap, data.dp, eos, DOF, f1, f2)
            return np.array([target_mdot - mdot, target_Tv  - Tv])
        f1, f2 = fsolve(func2, [100, 100])
        print(f1, f2)
        
        # def func2(x):
        #     f1 = x[0]
        #     mdot, Tv = solve_force_flux(Tliq, qvap, data.dp, eos, DOF, f1, f1)
        #     return target_Tv  - Tv # np.array([target_mdot - mdot])#, target_Tv  - Tv])
        # f1 = fsolve(func2, [1])
        # print(f1)
        
        # exit()
        # mdot, Tv = solve_force_flux(Tliq, qvap, data.dp, eos, DOF)#, method="RAUTER")
        # print(mdot, Tv-Tliq)
        # mdot, Tv = solve_force_flux(Tliq, qvap, data.dp, eos, DOF, f1=100, f2=100)
        # print(mdot, Tv-Tliq)
        # exit()
        # f1, f2 = 1e3, 1e3
        
        dp = np.linspace(0, -5, 15)
        qvapArr = np.linspace(0, -10, 20)
        mdot, Tv = np.zeros((15, 20)), np.zeros((15, 20))
        for i in range(len(dp)):
            for j in range(len(qvapArr)):
                mdot[i,j], Tv[i,j] = solve_force_flux(Tliq, qvapArr[j], dp[i], eos, DOF, f1=f1, f2=f2)#, method="RAUTER")
                # print(qvapArr[j], dp[i], mdot[i,j], Tv[i,j] - Tliq
        
        XX,YY = np.meshgrid(dp, qvapArr)
        
        c1 = a[0].contour(XX,YY,mdot.T, colors=("C0"))
        a[0].set_title(mdot_label)
        plt.clabel(c1, colors=("C0"), fmt='%1.2e')
        
        c2 = a[1].contour(XX,YY,Tv.T-Tliq, colors=("C3",))
        a[1].set_title('$T^g - T^\ell$ [K]')
        plt.clabel(c2, colors=("C3"))
        a[0].set_ylabel("$J_q^{'g}$ [W/m$^2$]")
        for a in axes:
            a.set_xlabel("$p - p_{sat}$ [Pa]")
        plt.tight_layout()
        plt.show()
        # # 
        # f1, f2 = 69.05435720501865, 115.98393038735588
        # f1, f2 = 55.7, 5.517
        # f1, f2 = 1e2, 1e2
        
        # mdot1, Tvap1= solve_force_flux(Tliq, qvap, data.dp, eos, DOF)
        # print(mdot1, Tvap1-Tliq)
        # mdot2, Tvap2= solve_force_flux(Tliq, qvap, data.dp, eos, DOF, f1, f2)
        # print(mdot2, Tvap2-Tliq)
        # mdot3, Tvap3= solve_force_flux(Tliq, qvap, data.dp, eos, DOF, f1=f1, f2=f2, method="RAUTER")
        # print(mdot3, Tvap3-Tliq)
        # mdot4, Tvap4 = data.mdot, 273.15 + data.T_gas[0]    # Jafari
        # sigma_c = KGT.sigma_condensation(data.T_liq[0] + 273.15, data.T_gas[0] + 273.15, DOF)
        # print(sigma_c, data.T_liq[0], data.T_gas[0], DOF)
        # j_hks = KGT.mass_flux_HKS(sigma_c, data.T_liq[0] + 273.15, data.T_gas[0] + 273.15, data.dp, eos)
        # print(j_hks)
        # labels = [r"Experiment \cite{jafariEvaporationMassFlux2018}", "KTG", "KTG scaled", "Linear fit"]
        # colors = ["C3", "C0", "C0", "C0"]
            
        # a[0].bar([1, 2, 3, 4, 5], [j_hks, mdot4, mdot1, mdot2, mdot3], color=["C4"] + colors, zorder=3)
        # a[1].bar([1, 2, 3, 4], [Tvap4-Tliq, Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq],  color=colors, zorder=3)
        
        # a[0].set_xticks([1, 2, 3, 4, 5])
        # a[0].set_xticklabels(["HKS"] + labels, rotation=45, ha='right')
        # a[0].set_ylabel(mdot_label)
        # a[1].set_xticks([1, 2, 3, 4])
        # a[1].set_xticklabels(labels, rotation=45, ha='right')
        # a[1].set_ylabel('$T^g - T^\ell$ [K]')
        # a[0].set_yscale('log')
        # for ax in a:
        #     ax.grid(axis='y', zorder=0)
        # plt.tight_layout()
        # plt.savefig('figure.pgf', backend='pgf')
        # plt.show()
    # elif eos == thermo.N2:
    #     DOF = thermo.DOF_N2
    #     Tliq = 77.3
    #     dp = -2
    #     qvap = -200
        
    #     target_mdot = 13.1e-3
    #     target_Tv = Tliq + 3.2
        
    #     # def func2(x):
    #     #     f1, f2 = x
    #     #     mdot, Tv = solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
    #     #     return np.array([target_mdot - mdot, target_Tv  - Tv])
    #     # f1, f2 = fsolve(func2, [1, 1])
        
    #     f1, f2 = 5e1, 2.5e5
    #     print(f1, f2)
    #     mdot1, Tvap1= solve_force_flux(Tliq, qvap, dp, eos, DOF)
    #     print(mdot1, Tvap1-Tliq)
    #     mdot2, Tvap2= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
    #     print(mdot2, Tvap2-Tliq)
    #     mdot3, Tvap3= solve_force_flux(Tliq, qvap, dp, eos, DOF, f1=1, f2=1, method="RAUTER")
    #     print(mdot3, Tvap3-Tliq)
    #     mdot4, Tvap4 = target_mdot, target_Tv    # Jafari
        
    #     labels = ["Scurlock", "KTG", "KTG (scaled)", "Rauter et al."]
    #     colors = ["C3", "C0", "C0", "C0"]
            
    #     a[0].bar([1, 2, 3, 4], [mdot4, mdot1, mdot2, mdot3], color=colors, zorder=3)
    #     a[1].bar([1, 2, 3, 4], [Tvap4-Tliq, Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq],  color=colors, zorder=3)
        
        # for ax in a:
        #     ax.set_xticks([1, 2, 3, 4])
        #     ax.set_xticklabels(labels, rotation=45, ha='right')
        #     ax.grid(axis='y', zorder=0)
        
    elif eos == thermo.H2:
        DOF = thermo.DOF_H2
        Tliq = eos.bubble_temperature(1e5, [1])[0]
        
        NASA_volume = 4700 #m^3
        NASA_R = (3*np.pi*NASA_volume/4)**(1/3)
        NASA_mass = 0.5*NASA_volume*thermo.calc_rho(Tliq, 1e5, eos.LIQPH, eos)
        target_mrate = 0.05*0.01*NASA_mass/(24*60*60)
        target_mdot = target_mrate/(np.pi*NASA_R**2)
        # print(target_mdot)
        # print(thermo.calc_T_sat(1.21e5, eos))
        # exit()
        # target_mdot = 13.1e-3
        # target_Tv = Tliq + 3.2
        
        # def func2(x):
        #     f1, f2 = x
        #     mdot, Tv = solve_force_flux(Tliq, qvap, dp, eos, DOF, f1, f2)
        #     return np.array([target_mdot - mdot, target_Tv  - Tv])
        # f1, f2 = fsolve(func2, [1, 1])
        
        # f1, f2 = 5e1, 3e5
        
        # pbpsat = np.linspace(0.99, 1, 20)
        # qvapArr = np.linspace(0, -10, 20)
        # mdot, Tv = np.zeros((20,20)), np.zeros((20,20))
        # for i in range(len(pbpsat)):
        #     for j in range(len(qvapArr)):
        #         mdot[i,j], Tv[i,j] = solve_force_flux(Tliq, qvapArr[j], pbpsat[i], eos, DOF, f1=f1, f2=f2)#, method="RAUTER")
        
        # XX,YY = np.meshgrid(pbpsat, qvapArr)
        # c1 = a[0].contour(XX,YY,mdot, colors=("C0"))
        # a[0].set_title("Mass flux")
        # plt.clabel(c1, colors=("C0"))
        # # plt.colorbar(c1)
        # c2 = a[1].contour(XX,YY,Tv-Tliq, colors=("C1",))
        # a[1].set_title("Temp. jump")
        # plt.clabel(c2, colors=("C1"))
        # # plt.colorbar(c2)
        # # plt.plot(qvapArr, mdot)
        # for a in axes:
        #     a.set_xlabel("$p/p_{sat}$")
        #     a.set_ylabel("q_{vap}")
        # plt.tight_layout()
        # plt.show()
        # exit()
        f1, f2 = 69.05435720501865, 115.98393038735588
        
        A, B = solve_force_flux(Tliq, -10, -5, eos, DOF)
        print(A, B - Tliq)
        
        dp = np.linspace(0, -5, 20)
        qvapArr = np.linspace(0, -10, 20)
        mdot, Tv = np.zeros((20,20)), np.zeros((20,20))
        for i in range(len(dp)):
            for j in range(len(qvapArr)):
                mdot[i,j], Tv[i,j] = solve_force_flux(Tliq, qvapArr[j], dp[i], eos, DOF, f1=10000, f2=10000)#, method="RAUTER")
        
        XX,YY = np.meshgrid(dp, qvapArr)
        
        m_frac = (60*60*24*mdot*np.pi*NASA_R**2)/NASA_mass
        
        c1 = a[0].contour(XX, YY, mdot.T, colors=("C0"))
        a[0].set_title(mdot_label)
        plt.clabel(c1, colors=("C0"), fmt='%1.2e')
        
        c2 = a[1].contour(XX,YY,Tv.T-Tliq, colors=("C3",))
        a[1].set_title('$T^g - T^\ell$ [K]')
        plt.clabel(c2, colors=("C3"), fmt='%1.4f')
        a[0].set_ylabel("$J_q^{'g}$ [W m$^{-2}$]")
        # for a in axes:
        for ax in a:
            ax.set_xlabel("$p - p_{sat}$ [Pa]")
        plt.tight_layout()
        plt.show()
        
        # print(f1, f2)
        # mdot1, Tvap1= solve_force_flux(Tliq, -0.1, -0.1, eos, DOF)#, f1, f2)
        # print(mdot1, Tvap1-Tliq)
        # mdot2, Tvap2= solve_force_flux(Tliq, -1, 0.99, eos, DOF, f1, f2)
        # print(mdot2, Tvap2-Tliq)
        # mdot3, Tvap3= solve_force_flux(Tliq, -1, 0.9, eos, DOF, f1, f2)
        # print(mdot3, Tvap3-Tliq)
        # mdot4, Tvap4 = target_mdot, target_Tv    # Jafari
        
        # labels = ["KTG (scaled)", "Rauter et al."]
        # colors = ["C0", "C0", "C0"]
            
        # plt.bar([1, 2, 3], [mdot1, mdot2, mdot3], color=colors, zorder=3)
        # a[1].bar([1, 2, 3], [Tvap1-Tliq, Tvap2-Tliq , Tvap3-Tliq],  color=colors, zorder=3)
        
        # for ax in a:
        #     ax.set_xticks([1, 2, 3, 4])
        #     ax.set_xticklabels(labels, rotation=45, ha='right')
        #     ax.grid(axis='y', zorder=0)
            
        
    # a[0].set_yscale("log")
    # a[0].set_ylabel(r"$\dot{m}$ [kg/s]")
    # a[1].set_ylabel("$T^g - T^\ell$ [K]")
    # fig.tight_layout()
    # plt.show()


