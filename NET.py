import numpy as np

#################################################################
#                                                               #
#               Linear fits from Rauter et al.                  #
#           Only valid for water at low pressures!              #
#                                                               #
def r_qq(Tliq):                                                 #
    return -1.5636e-8*Tliq + 4.6189e-6                          #
#                                                               #
def r_qmu(Tliq):                                                #
    return -0.0026*Tliq + 0.7415                                #
#                                                               #
def r_mumu(Tliq):                                               #
    return -2.7399e-3*Tliq + 8.0423e5                           #
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
    plt.style.use(["science", "nature"])
    Tl = np.linspace(260, 280)
    
    rqq_NET = r_qq(Tl)
    rmuq_NET = r_qmu(Tl)
    rmumu_NET = r_mumu(Tl)
    
    Ceq = thermo.calc_Ceq(Tl, thermo.water)
    rqq = KGT.R_qq(Ceq, Tl, thermo.R, thermo.M_water)
    rmuq = KGT.R_qmu(Ceq, Tl, thermo.R, thermo.M_water)
    # rmumu = KGT.R_mumu(Ceq, Tl, thermo.R, thermo.M_water, Tl, thermo.DOF_water, sigma=False)
    rmumu_simga0_1 = KGT.R_mumu(Ceq, Tl, thermo.R, thermo.M_water, Tl, thermo.DOF_water, sigma=0.1)
    rmumu_simga0_9 = KGT.R_mumu(Ceq, Tl, thermo.R, thermo.M_water, Tl, thermo.DOF_water, sigma=0.9)

    fig, axes = plt.subplots(3, 1, sharex=True)
    a = axes.flatten()
    c2 = "C3"
    a[0].plot(Tl, rqq, label="KTG")
    a[0].plot(Tl, rqq_NET, color=c2)
    a[0].set_ylabel("$R_{qq}^{s,g}$ [m$^2$s/(J$\cdot$K)]")
    a[0].set_ylim((0.5e-8, 1e-6))
    a[1].plot(Tl, rmuq)
    a[1].plot(Tl, rmuq_NET, color=c2)
    a[1].set_ylabel("$R_{q\mu}^{s,g}$ [m$^2$s/(mol$\cdot$K)]")
    a[2].plot(Tl, rmumu_simga0_1, label="KTG ($\sigma_c = 0.1$)", color="C0")
    a[2].plot(Tl, rmumu_simga0_9, label="KTG ($\sigma_c = 0.9$)", ls='--', color="C0")
    a[2].plot(Tl, rmumu_NET, color=c2, label="Rauter et al.")
    a[2].set_ylabel("$R_{\mu\mu}^{s,g}$ [m$^2$s/(mol$^2\cdot$K)]")
    a[2].set_ylim((1e-2, 1e7))

    a[-1].set_xlabel("T [K]")
    for ax in a:
        ax.grid()
        ax.set_yscale("log")
    plt.tight_layout()
    
    legend = a[2].legend(framealpha=0.0, facecolor='white', edgecolor='black')
    # legend.get_frame().set_alpha(None)
    plt.show()
    
