import numpy as np

def mass_flux_NET(T_g, T_l, p_g, p_sat, R, M, Lww, Lwq):
    P1 = -Lww*(R*T_l/M)*np.log(p_g/p_sat)
    P2 = -Lwq*(T_g - T_l)/T_g
    return P1 + P2

def heat_flux_gas_NET(T_g, T_l, p_g, p_sat, R, M, Lqq, Lwq):
    P1 = -Lwq*(R*T_l/M)*np.log(p_g/p_sat)
    P2 = -Lqq*(T_g - T_l)/T_g
    return P1 + P2