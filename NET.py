import numpy as np

#################################################################
#                                                               #
#               Linear fits from Rauter et al.                  #
#           Only valid for water at low pressures!              #
# 
Ts_water = 647.096
def r_qq(Tliq, Ts):                                             #
    return (-1.5636e-8)*(Tliq/Ts)*Ts_water + 4.6189e-6            #
#                                                               #
def r_qmu(Tliq, Ts):                                                #
    return -0.0025*(Tliq/Ts)*Ts_water + 0.7415                                #
#                                                               #
def r_mumu(Tliq, Ts):                                               #
    return (-2.7399e3)*(Tliq/Ts)*Ts_water + 8.0423e5                           #
#                                                               #
#                                                               #
#################################################################


def forces(q_g, J, rqq, rmumu, rqmu):
    """
    Force-flux equations for steady evaporation in NET, single component
    See Eq. (11.14) in Non-Equilibrium Thermodynamics of Heterogeneous Systems
    by Kjelstrup and Bedeaux, 2. edition.

    Args:
        q_g (float): heat flux on gas side of interface    [W/m^2]
        J (float): molar flux across interface             [mol/(m^2 s)]
        rqq (float):   Heat resistance coeff               [m^2 s / (J K)]
        rmumu (float): Mass resistance coeff               [J m^2 s / (mol^2 K)]
        rqmu (float):  Coupling coeff                      [m^2 s / (mol K)]

    Returns: Resulting forces
        (float, float): (1/T_g - 1/T_l), -R*ln(p/p_sat)    [1/K], [J/(mol K)]
    """
    return rqq*q_g + rqmu*J, rqmu*q_g + rmumu*J


if __name__ == "__main__":
    import KGT, thermo
    
    import matplotlib.pyplot as plt
    # Updating plt.rcParams for the legend
    import scienceplots
    plt.style.use(["science", "bright", "nature"])
    Tl = np.linspace(260, 280)
    
    # eos = thermo.H2
    # M = thermo.M_H2
    # DOF = thermo.DOF_H2
    # Tc = thermo.Tc_H2
    
    eos = thermo.water
    M = thermo.M_water
    DOF = thermo.DOF_water
    Tc = thermo.Tc_water
    
    rqq_NET = r_qq(Tl, Tc)
    rmuq_NET = r_qmu(Tl, Tc)*M
    rmumu_NET = r_mumu(Tl, Tc)*M**2
    
    rqq_NET_red = r_qq(Tl, Tc)
    rmuq_NET_red = r_qmu(Tl, Tc)*M
    rmumu_NET_red = r_mumu(Tl, Tc)*M**2
    
    Ceq = thermo.calc_Ceq(Tl, eos)
    rqq = KGT.R_qq(Ceq, Tl, thermo.R, M)
    rmuq = KGT.R_qmu(Ceq, Tl, thermo.R, M)
    # rmumu = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=False)
    rmumu_simga0_1 = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=0.1)
    rmumu_simga0_9 = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=0.9)
    
    f1, f2 = 69.05435720501865, 115.98393038735588
    
    rqq_scaled = KGT.R_qq(Ceq, Tl, thermo.R, M)*f1
    rmuq_scaled = KGT.R_qmu(Ceq, Tl, thermo.R, M)*f1
    # rmumu = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=False)
    rmumu_simga0_1_scaled = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=0.1)*f2
    rmumu_simga0_9_scaled = KGT.R_mumu(Ceq, Tl, thermo.R, M, Tl, DOF, sigma=0.9)*f2

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(5,4))
    a = axes.flatten()
    c2 = "C1"
    c_scaled = "C0"
    c_KTG = "k"
    ms_09 = "o"
    ms_01 = "^"
    me = 8
    a[0].plot(Tl, rqq, label="KTG", color=c_KTG)
    a[0].plot(Tl, rqq_scaled, label="KTG Scaled", color=c_scaled)
    a[0].plot(Tl, rqq_NET, color=c2)
    # a[0].plot(Tl, rqq_NET_red, color=c2, ls='--')
    a[0].set_ylabel("$R_{qq}^{g}$ [m$^2$s/(J$\cdot$K)]")
    a[0].set_yscale("symlog",linthresh=1e-12)
    a[0].set_yticks([1e-5, 1e-7, 1e-9, 1e-11, 0, -1e-11, -1e-9, -1e-7, -1e-5])
    # a[0].set_ylim((0.5e-8, 1e-6))
    a[1].plot(Tl, rmuq, color=c_KTG)
    a[1].plot(Tl, rmuq_NET, color=c2)
    a[1].plot(Tl, rmuq_scaled, color=c_scaled)
    # a[1].plot(Tl, rmuq_NET_red, color=c2, ls='--')
    a[1].set_ylabel("$R_{q\mu}^{g}$ [m$^2$s/(mol$\cdot$K)]")
    a[1].set_yscale("symlog",linthresh=1e-12)
    a[1].set_yticks([1e-5, 1e-7, 1e-9, 1e-11, 0, -1e-11, -1e-9, -1e-7, -1e-5])
    a[2].plot(Tl, rmumu_simga0_1, color=c_KTG, marker=ms_01, markevery=me)
    a[2].plot(Tl, rmumu_simga0_1_scaled, color=c_scaled, marker=ms_01, markevery=me)
    a[2].plot(Tl, rmumu_simga0_9, marker=ms_09, color=c_KTG, markevery=me)
    a[2].plot(Tl, rmumu_simga0_9_scaled, marker=ms_09, color=c_scaled, markevery=me)
    a[2].plot(Tl, rmumu_NET, color=c2, label="Linear fit")
    # a[2].plot(Tl, rmumu_NET_red, color=c2, label="Reduced Rauter et al.", ls='--')
    a[2].set_ylabel("$R_{\mu\mu}^{g}$ [m$^2$s/(mol$^2\cdot$K)]")
    # a[2].set_ylim((1e-2, 1e7))
    a[2].plot([],[],label=f"$\sigma_c = 0.9$", color='grey', marker=ms_09)
    a[2].plot([],[],label=f"$\sigma_c = 0.1$", color='grey', marker=ms_01)
    
    a[3].plot(Tl, rqq*rmumu_simga0_1 - rmuq**2, color=c_KTG, marker=ms_01, markevery=me)
    a[3].plot(Tl, rqq*rmumu_simga0_9 - rmuq**2, color=c_KTG, marker=ms_09, markevery=me)
    a[3].plot(Tl, rqq_scaled*rmumu_simga0_1_scaled - rmuq_scaled**2, color=c_scaled, marker=ms_01, markevery=me)
    a[3].plot(Tl, rqq_scaled*rmumu_simga0_9_scaled - rmuq_scaled**2, color=c_scaled, marker=ms_09, markevery=me)
    a[3].plot(Tl, rqq_NET*rmumu_NET - rmuq_NET**2, color=c2)
    # a[3].plot(Tl, rqq_NET_red*rmumu_NET_red - rmuq_NET_red**2, color=c2, label="Reduced Rauter et al.", ls='--')
    a[3].set_yscale("symlog",linthresh=1e-12)
    a[3].set_ylabel(r"$R_{\mu\mu}^{g}R_{qq} - {R_{q\mu}^{g}}^2$ [m$^4$s$^2$/(mol$^4\cdot$K$^2$)]")
    # a[2].set_ylabel("$R_{\mu\mu}^{s,g}$ [m$^2$s/(mol$^2\cdot$K)]")
    

    a[-1].set_xlabel(r"$T^\ell$ [K]")
    a[-2].set_xlabel(r"$T^\ell$ [K]")
    for ax in a:
        ax.grid()
        ax.set_yscale("log")
    plt.tight_layout()
    
    legend = fig.legend(loc='lower center', ncol=5, borderpad=0)
    # legend.get_frame().set_alpha(None)
    plt.savefig("coeffs_water.pdf")
    plt.show()
    
