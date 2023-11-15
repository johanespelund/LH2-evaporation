import numpy as np

def mass_flux_NET(T_g, T_l, p_g, p_sat, R, M, Lww, Lwq):
    P1 = -Lww*(R*T_l/M)*np.log(p_g/p_sat)
    P2 = -Lwq*(T_g - T_l)/T_g
    return P1 + P2

def heat_flux_gas_NET(T_g, T_l, p_g, p_sat, R, M, Lqq, Lwq):
    P1 = -Lwq*(R*T_l/M)*np.log(p_g/p_sat)
    P2 = -Lqq*(T_g - T_l)/T_g
    return P1 + P2

def calc_Lqq(*args):
    pass


def deltaT_inv(q_l, mdot, rqq_sl, rqmu_sl):
    return rqq_sl*q_l + rqmu_sl*mdot

def p_by_psat(q_l, mdot, rmumu_sl, rqmu_sl, R):
    return -(rqmu_sl*q_l + rmumu_sl*mdot)/R

def R_qq(C_eq, Tl, R, M):
    v_mp = np.sqrt((2*R*Tl)/M)     # m/s
    return ((np.sqrt(np.pi))/(4 * C_eq* R * Tl**2 * v_mp))*(1 + 104/(25*np.pi))

def R_qmu(C_eq, Tl, R, M):
    v_mp = np.sqrt((2*R*Tl)/M)     # m/s
    return ((np.sqrt(np.pi))/(8 * (C_eq) * Tl * v_mp))*(1 + 16/(5*np.pi))

def R_mumu(C_eq, Tl, R, M):
    v_mp = np.sqrt((2*R*Tl)/M)     # m/s
    return ((2 * (R)) * np.sqrt(np.pi))/((C_eq) * v_mp) * ((0.9)**(-1) + np.pi**(-1) - 23/32)


if __name__ == "__main__":
    from thermo import calc_p_sat, water, M_water, calc_Ceq
    import matplotlib.pyplot as plt
    R = 8.314
    Tl = np.linspace(230, 300)
    psat = calc_p_sat(Tl, water)
    # vg, = water.specific_volume(
    #     Tl, psat, [1], water.VAPPH)
    C_eq = calc_Ceq(Tl, water) 
    # print(C_eq)
    # print(psat*1e-5)
    rqq = R_qq(C_eq, Tl, R, M_water)
    rmuq = R_qmu(C_eq, Tl, R, M_water)
    rmumu = R_mumu(C_eq, Tl, R, M_water)

    fig, axes = plt.subplots(2, 2)
    a = axes.flatten()
    a[0].plot(Tl, psat)
    a[1].plot(Tl, rqq)
    a[2].plot(Tl, rmuq)
    a[3].plot(Tl, rmumu)

    plt.show()

    print(f"{rqq=}")
    print(f"{rmuq=}")
    print(f"{rmumu=}")
    
