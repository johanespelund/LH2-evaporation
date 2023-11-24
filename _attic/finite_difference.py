import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from _attic.density_func import density

N = 25
L = 1

# Constants
R = 287.05  # J/(kg*K), specific gas constant for air
c_p = 1005  # J/(kg*K), specific heat at constant pressure for air
Q_dot = 1000  # W/m, constant heat source term
kappa = 0.003           # [W/(m*K)]
liquid_density = 1000

# Boundary conditions
T_inlet = 280  # K, inlet temperature
m_dot = 0.01   # kg/(m^2*s), mass flux
p_inlet = 101325  # Pa, outlet pressure

# Inlet density from ideal gas law
rho_inlet = liquid_density #p_inlet / (R * T_inlet)

# Inlet mass flow rate
u_inlet = m_dot/rho_inlet  # m/s

x = np.linspace(-L/2, L/2, N)

non_linear_points = np.linspace(-np.sqrt(L/2), 0, N)**2
dense_points_right = np.flip(non_linear_points)
dense_points_left = -non_linear_points

dense_points = np.concatenate((dense_points_left[:-1], dense_points_right))
x = dense_points

dx = np.gradient(x)

u = np.ones(x.size)*u_inlet
p = np.ones(x.size)*p_inlet
h = np.ones(x.size)*T_inlet*c_p
rho = np.ones(x.size)*rho_inlet
T = np.ones(x.size)*T_inlet
Jq = np.zeros(x.size)

def Q_dot(T):
    T_ref = 300
    HTC = 100
    return HTC*(T_ref - T)

mass_error = []
momentum_error = []
energy_error = []

N_iter = 50

for _ in range(N_iter):
    dudx = np.gradient(u, x, edge_order=2)
    Jq = -kappa*np.gradient(T, x, edge_order=2)
    dhdx = (-np.gradient(Jq, x, edge_order=2) + Q_dot(T))/(m_dot)
    dpdx = -m_dot*dudx
    
    # Use marching method to integrate ODE's
    def dhdx(h, xx):
        interp = np.interp(xx, x, (-np.gradient(Jq, x, edge_order=2) + Q_dot(T))/(m_dot))
        return interp
    
    def dpdx(p, xx):
        interp = np.interp(xx, x, (-m_dot*dudx))
        return interp
        
    h = odeint(dhdx, h[0], x).T[0]
    p = odeint(dpdx, p[0], x).T[0]
    
    # Update temperature, density and velocity
    T = h/c_p
    rho = density(x, T, p, R, liquid_density, 0.01)
    u = m_dot/rho
    
    # Check conservation
    
    mass_flux = np.gradient(u*rho, x, edge_order=2)
    momentum_flux = np.gradient(rho*u**2 + p, edge_order=2)
    energy_flux = (np.gradient(rho*u*h, edge_order=2) + np.gradient(-kappa*np.gradient(T, x, edge_order=2), edge_order=2) - Q_dot(T))
    
    mass_balance = mass_flux/(m_dot/L)
    momentum_balance = momentum_flux/(m_dot*u_inlet + p_inlet/L)
    energy_balance = energy_flux/(m_dot*h[0]/L)
    
    mass_error.append(np.linalg.norm(mass_balance))
    momentum_error.append(np.linalg.norm(momentum_balance))
    energy_error.append(np.linalg.norm(energy_balance))

print(f"mass error = {mass_error[0]}")
print(f"momentum error = {momentum_error[0]}")
print(f"energy error = {energy_error[0]}")
   

fig, axes = plt.subplots(2, 2)
a = axes.flatten()

a[0].plot(x, T)
a[0].set_ylabel("Temperature [K]")
   
a[1].plot(x, p*1e-5)
a[1].set_ylabel("Pressure [bar]")
   
a[2].plot(x, u)
a[2].set_ylabel("Velocity [m/s]")

a[3].plot(x, rho)
a[3].set_ylabel("Density [kg/m^3]")

fig2, axes2 = plt.subplots(2, 2)
a2 = axes2.flatten()

a2[0].plot(x, energy_balance, ls='--')
a2[0].set_ylabel("Energy flux gradient[W/m^3]")
   
a2[1].plot(x, momentum_balance, ls='--')
a2[1].set_ylabel("Momentum flux gradient[kg/(m^2*s^2)]")
   
a2[2].plot(x, mass_balance, ls='--')
a2[2].set_ylabel("Mass flux gradient [kg/(m^3*s]")

a2[3].plot(np.arange(N_iter), mass_error, label="Mass")
a2[3].plot(np.arange(N_iter), momentum_error, label="Momentum")
a2[3].plot(np.arange(N_iter), energy_error, label="Energy")
a2[3].set_yscale('log')
a2[3].set_ylabel('Normalized error')
a2[3].set_xlabel('Iteration number')
a2[3].legend()

for ax in list(a)+list(a2):
    ax.grid()

fig.tight_layout()
fig2.tight_layout()
plt.show()


# plt.figure(1)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
