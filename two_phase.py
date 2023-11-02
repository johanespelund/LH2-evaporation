import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from thermopack.cubic import cubic
from thermopack.cpa import cpa
from thermopack import thermo

from density_func import density
from dataclasses import dataclass

N = 15
L = 1

x = np.linspace(-L/2, 0, N)

non_linear_points = np.linspace(-np.sqrt(L/2), 0, N)**2
dense_points_right = np.flip(non_linear_points)
dense_points_left = -non_linear_points

dense_points = np.concatenate((dense_points_left[:-1], dense_points_right))
# x = dense_points
N = x.size

x_liq = x
x_vap = -np.flip(x)

# print(x_vap, x_liq)
# exit()

# Constants
R = 287.05  # J/(kg*K), specific gas constant for air
c_p = 1005  # J/(kg*K), specific heat at constant pressure for air
kappa = 0.062           # [W/(m*K)]
liquid_density = 1000

htc = 10
T_ref = 300

# Boundary conditions
T_inlet = 295           # K, inlet temperature
m_dot = 0.1            # kg/(m^2*s), mass flux
p_inlet = 1.01325e5     # Pa, inlet pressure

# Inlet density from ideal gas law
rho_inlet = liquid_density #p_inlet / (R * T_inlet)
u_inlet = m_dot/rho_inlet  # m/s

water = cpa('H2O', 'SRK')
h_inlet =  water.enthalpy(T_inlet, p_inlet, [1], water.LIQPH)[0]
flsh = water.two_phase_phflash(p_inlet, [1], h_inlet)
print(h_inlet, flsh.T)

class Phase:
    def __init__(self, x, N, eos, phase_flag):
        self.x = x
        self.N = N
        self.eos = eos
        self.phase_flag = phase_flag
        self.u = np.ones(N) * u_inlet
        self.rho = np.ones(N) * rho_inlet
        self.h = np.ones(N) * h_inlet
        self.T = np.ones(N) * T_inlet
        self.p = np.ones(N) * p_inlet
        self.q = np.zeros(N)
        
        # Initialize gradients
        self.dhdx = np.ones(N)
        self.dTdx = np.ones(N)
        self.dpdx = np.ones(N)
        
        # Additional initializations can be called here as they were in __post_init__
        self.dx = np.gradient(self.x)
        self.M = self.eos.compmoleweight(1) * 1e-3  # [kg/mol]
        self.fields = [self.u, self.rho, self.h, self.T, self.p, self.q]
    
    # Calculation of gradients
    def calc_dhdx(self):
        self.dhdx = (-np.gradient(self.q, self.x, edge_order=2) + self.Q_dot(htc, T_ref))/m_dot 
        return self.dhdx
    
    def calc_dpdx(self):
        self.dpdx = -m_dot*self.dudx
        return -m_dot*self.dudx
    
    def update_fields(self):
        self.dudx = np.gradient(self.u, self.x, edge_order=2)
        self.q = -kappa*np.gradient(self.T, self.x, edge_order=2)
        self.calc_dhdx()
        self.calc_dpdx()
        
        # Integrate to find profiles
        self.h[1:] = self.h[:-1] + self.dhdx[:-1]*self.dx[:-1]
        self.p[1:] = self.p[:-1] + self.dpdx[:-1]*self.dx[:-1]
        
        # Update temperature, density and velocity
        # self.T = self.h/c_p
        self.calc_temp()
        self.calc_density()
        self.u = m_dot/self.rho
        
    def Q_dot(self, htc, T_ref):
        return htc*(T_ref - self.T)
    
    def calc_density(self):
        # Assume liquid for now
        for i in range(self.N):
            specific_volume, = self.eos.specific_volume(self.T[i], self.p[i], [1], self.phase_flag) # [m^3/mol]
            self.rho[i] = self.M/specific_volume
    
    def calc_kappa(self):
        # Assume liquid for now
        for i in range(self.N):
            specific_volume, = self.eos.tc(self.T[i], self.p[i], [1], self.phase_flag) # [m^3/mol]
            self.rho[i] = self.M/specific_volume
    
    def calc_temp(self):
        for i in range(self.N):
            flsh = self.eos.two_phase_phflash(self.p[i], [1], self.h[i])
            self.T[i] = flsh.T

liq = Phase(x=x_liq, N=x_liq.size, eos=water, phase_flag=water.LIQPH)
vap = Phase(x=x_vap, N=x_vap.size, eos=water, phase_flag=water.VAPPH)

N_iter = 100

for _ in range(N_iter):
    liq.update_fields()
    for i in range(len(liq.fields)):
        vap.fields[i][0] = liq.fields[i][-1]
    dH_vap = water.enthalpy(vap.T[0], vap.p[0], [1], water.VAPPH)[0] - water.enthalpy(vap.T[0], vap.p[0], [1], water.LIQPH)[0]
    print(dH_vap)
    vap.h[0] = liq.h[-1] + dH_vap
    vap.calc_temp()
    vap.update_fields()
    
        
fig, axes = plt.subplots(2, 2)
a = axes.flatten()

for phase in [liq, vap]:
    a[0].plot(phase.x, phase.T)
    a[1].plot(phase.x, phase.p) 
    a[2].plot(phase.x, phase.u)
    a[3].plot(phase.x, phase.rho)
    
a[0].set_ylabel("Temperature [K]")
a[1].set_ylabel("Pressure [bar]")
a[2].set_ylabel("Velocity [m/s]")
a[3].set_ylabel("Density [kg/m^3]")

plt.show()