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
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(["science", "nature"])
    Tl = np.linspace(273, 300)
    
    rqq = r_qq(Tl)
    rmuq = r_qmu(Tl)
    rmumu = r_mumu(Tl)

    fig, axes = plt.subplots(1, 3)
    a = axes.flatten()
    a[0].plot(Tl, rqq)
    a[0].set_ylabel("$R_{qq}^{s,g}$ [m$^2$s/(J$\cdot$K)]")
    a[1].plot(Tl, rmuq)
    a[1].set_ylabel("$R_{q\mu}^{s,g}$ [m$^2$s/(mol$\cdot$K)]")
    a[2].plot(Tl, rmumu)
    a[2].set_ylabel("$R_{\mu\mu}^{s,g}$ [m$^2$s/(mol$^2\cdot$K)]")

    
    for ax in a:
        ax.grid()
        ax.set_xlabel("T [K]")
    plt.tight_layout()
    plt.show()
    print(f"{rqq=}")
    print(f"{rmuq=}")
    print(f"{rmumu=}")
    
