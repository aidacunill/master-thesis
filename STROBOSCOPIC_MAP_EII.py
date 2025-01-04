#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Let us UPLOAD the desired libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import math
from matplotlib.animation import FuncAnimation
np.set_printoptions(precision=15, suppress=False) #stablish the output number of decimals
import scipy.special as sp
import scipy.signal as signal
from scipy.optimize import root
get_ipython().system('pip install numdifftools #necessary to compute numerical derivatives')


# ## Computation of Neural System ODEs: E-I-I with an excitatory input

# In[2]:


# The parameters we consider are as follows. For their biological significance, please refer to the thesis.
tau_e = 8
tau_i = 8
tau_2=8
tau_se = 1
tau_si = 5
tau_s2 = 30
Jee = 0
Jii = 0
J22 = 0
Jie = 13
Jei = 13
J2e = 13
Je2 = 13
Delta_e = 1
Delta_i = 1
Delta_2 = 1
eta_e = -5
eta_i = -5
eta_2 = -5
I_bar_e = 10
I_bar_i = 0
I_bar_2 = 0

# Initial condition: Taken from "Communication through coherence in a realistic neuronal model" by David Reyner Parra.
x0_david = [0.085262756825722,
1.125737037325799,
0.000000000000000,
0.209748707974244,
0.012828206758546,
-1.136491592341226,
0.834230210263032,
0.000000000000000]

x0_david_I = [0.085262756825722,
1.125737037325799,
0.000000000000000,
0.209748707974244,
0.012828206758546,
-1.136491592341226,
0.834230210263032,
0.000000000000000,
0.012828206758546,
-1.136491592341226,
0.209748707974244,
0.834230210263032,0]

# Neural network system of ODEs for the unperturbed system. 
#For details on their biological significance, please refer to the thesis.
def neural_network_notperturbed(x, t):

   # Compute the total external input to the neurons
    I_e = I_bar_e + tau_e * x[2] - tau_e * x[3]-tau_e*x[10]
    I_i = I_bar_i + tau_i * x[6] - tau_i * x[7]
    I_2 = I_bar_2 + tau_2 * x[11] - tau_2 * x[12]

    # System of equations (neuronal dynamics)
    dxdt = [
        (1 / tau_e) * (Delta_e / (np.pi * tau_e) + 2 * x[0] * x[1]),  # re
        (1 / tau_e) * (x[1] ** 2 + eta_e + I_e - (tau_e * np.pi * x[0]) ** 2),  # Ve
        (1 / tau_se) * (-x[2] + Jee * x[0]),  # See
        (1 / tau_si) * (-x[3] + Jei * x[4]),  # Sei
        (1 / tau_i) * (Delta_i / (np.pi * tau_i) + 2 * x[4] * x[5]),  # ri
        (1 / tau_i) * (x[5] ** 2 + eta_i + I_i - (tau_i * np.pi * x[4]) ** 2),  # Vi
        (1 / tau_se) * (-x[6] + Jie * x[0]),  # Sie
        (1 / tau_si) * (-x[7] + Jii * x[4]),   # Sii
        (1 / tau_2) * (Delta_2 / (np.pi * tau_2) + 2 * x[8] * x[9]),  # r2
        (1 / tau_2) * (x[9] ** 2 + eta_2 + I_2 - (tau_2 * np.pi * x[8]) ** 2),  # V2
        (1 / tau_s2) * (-x[10] + Je2 * x[8]),  # Se2
        (1 / tau_se) * (-x[11] + J2e * x[0]),  # S2e
        (1 / tau_s2) * (-x[12] + J22 * x[8])]  # S22
    return dxdt
    
    


# In[3]:


# The following code implements the Poincaré section. 
#The theoretical background is explained in the appendix of the thesis.

# g(x) function which defines the Poincaré section
def g(x):
    return x[1]+0.5

def approximation_tau_dir(system, x0, g, atol, rtol, dir, plot, estimated_period):
    # Ensure t is strictly monotonic
    if dir == 1:
        t = np.linspace(0, estimated_period, 5000)
    elif dir == -1:
        t = np.linspace(0, -estimated_period, 5000)  # Reverse for monotonicity
    
    # Solve the ODE
    sol_simp = odeint(system, x0, t, atol=atol, rtol=rtol)
    x_i = sol_simp[1]
    
    # Find crossing of Poincaré section
    for i, xi in enumerate(sol_simp[1:], start=1):  # Avoid first element
        if g(x_i) * g(xi) <= 0:  # Check for sign change
            t0 = t[i]  # Time of crossing
            break
        x_i = xi  # Update x_i

    if plot == 1:
        # Plot the solution in the phase plane
        plt.figure(figsize=(4, 4))
        plt.plot(sol_simp[:, 0], sol_simp[:, 1])
        plt.xlabel('$r_e$ (kHz)')
        plt.ylabel('$V_e$ (mV)')
        plt.title('Periodic Oscillator')
        plt.grid(True)
        plt.show()

    return [xi, t0]

# One iteration of Newton's method
def DeltaT(x0):
    I_e = I_bar_e + tau_e * x0[2] - tau_e * x0[3]-tau_e*x0[10]
    Ve=(1/tau_e) * (x0[1]**2 + eta_e + I_e - (tau_e * np.pi * x0[0])**2)
    return - g(x0) / Ve

# Function to evaluate the system's solution at a given direction and time
def evaluate(system, x0, h, dir, atol, rtol):
    if dir == 1: t = np.linspace(0, h, 100)  # Small time step. Forward integration
    if dir == -1 : t = np.linspace(0, -h, 100) # Backward integration
    sol = odeint(system, x0, t, atol=atol, rtol=rtol)
    return sol[-1]

def poincare_map_n_periodic(system, x0,n, tol, atol, rtol,dir,plot,estimated_period):
    points = np.zeros((n, 13))  # To store intersection points
    X0=x0
    total_time = 0
    for i in range(n):
        # Use last intersection point as new initial condition
      if i != 0: x0 = xk
      approx = approximation_tau_dir(neural_network_notperturbed,x0,g,atol,rtol,dir,plot,estimated_period)
      xk = approx[0]
      total_time += approx[1]

      # Refine the intersection using Newton's method
      while abs(g(xk)) > tol:
            deltaT = DeltaT(xk)
            total_time += deltaT
            if deltaT < 0:
                  xk = evaluate(system, xk, abs(deltaT), -1, atol, rtol)
            else:
                  xk = evaluate(system, xk, abs(deltaT), 1, atol, rtol)
      points[i, :] = xk  # Store the refined intersection point

    return points,total_time, points[-1]


# In[4]:


t = np.linspace(0, 550, 2500)

# Parameters
tol = 1e-16  # Tolerance for Newton's method
atol = 1e-17  # Absolute tolerance for ODE integration
rtol = 1e-13  # Relative tolerance for ODE integration

sol = odeint(neural_network_notperturbed, x0_david_I, t, atol=atol, rtol=rtol)
plt.plot(sol[:,0],sol[:,1])


# In[5]:


t = np.linspace(0, 1000, 25000)

# Parameters
tol = 1e-13  # Tolerance for Newton's method
atol = 1e-17  # Absolute tolerance for ODE integration
rtol = 1e-13  # Relative tolerance for ODE integration

sol = odeint(neural_network_notperturbed, x0_david_I, t, atol=atol, rtol=rtol)

# Time threshold to avoid the initial transient (e.g., 50 time units)
time_threshold = 800  # Time threshold in the simulation

# Only consider peaks that occur after the time threshold
valid_times = t[t >= time_threshold]
valid_r_e = sol[t >= time_threshold, 0]

# Find the peaks in the valid portion of r_e
peaks, _ = signal.find_peaks(valid_r_e)
# Compute the period by calculating the time difference between consecutive peaks
periods = np.diff(valid_times[peaks])  # Differences between consecutive peaks

# The estimated period
estimated_period = np.mean(periods)  # Average period
print(estimated_period)
x0=sol[-1]

t = np.linspace(0, 100, 2000)
sol = odeint(neural_network_notperturbed, x0, t, atol=atol, rtol=rtol)
plt.plot(sol[:,0],sol[:,1])

P_0= poincare_map_n_periodic(neural_network_notperturbed, x0,1, tol, atol, rtol, dir=1,plot=0,estimated_period=estimated_period)[2]
P_2= poincare_map_n_periodic(neural_network_notperturbed, P_0,1,1e-14, atol, rtol, dir=1,plot=0,estimated_period=estimated_period)[2]
plt.scatter(P_0[0],P_0[1],color='red')
plt.scatter(P_2[0],P_2[1],color='green')
plt.show()


# In[6]:


# The following code is used to find a point that belongs to the oscillator and calculate its period T*.

def Poincare(x):
    result = poincare_map_n_periodic(
        neural_network_notperturbed,
        x,
        n=2,
        tol=tol,
        atol=atol,
        rtol=rtol,
        dir=1,
        plot=0,
        estimated_period=estimated_period
    )[2]-x
    return result

# Find the root of the Poincaré map
result = root(Poincare, P_0,method='hybr')

# Check and display the result of the root-finding process
if result.success:
    print(f"Root found: {result.x}")
else:
    print("No root found.")

P_1=result.x
Period = poincare_map_n_periodic(neural_network_notperturbed,P_1 ,2, tol, atol, rtol, dir=1,plot=0,estimated_period=estimated_period)[1]
print(f"Period of the oscillator, T* = {Period}")


# In[7]:


#Graphic representation of the oscillator
t = np.linspace(0, Period, 500)

# Parameters
tol = 1e-16  # Tolerance for Newton's method
atol = 1e-17  # Absolute tolerance for ODE integration
rtol = 1e-13  # Relative tolerance for ODE integration

sol = odeint(neural_network_notperturbed, P_1, t, atol=atol, rtol=rtol)

# Create a figure and primary axis
fig, ax1 = plt.subplots(figsize=(6, 6))

# Plot variables on the primary axis
ax1.plot(t, sol[:, 1], label=r'$V_e$', color='tab:blue')
ax1.plot(t, sol[:, 2], label=r'$S_{ee}$', color='tab:orange')
ax1.plot(t, sol[:, 3], label=r'$S_{ei}$', color='tab:green')
ax1.plot(t, sol[:, 5], label=r'$V_i$', color='tab:red')
ax1.plot(t, sol[:, 6], label=r'$S_{ie}$', color='tab:purple')
ax1.plot(t, sol[:, 7], label=r'$S_{ii}$', color='tab:brown')
ax1.plot(t, sol[:, 9], label=r'$V_2$', color='tab:red')
ax1.plot(t, sol[:, 10], label=r'$S_{e2}$', color='black')
ax1.plot(t, sol[:, 11], label=r'$S_{2e}$', color='yellow')
ax1.plot(t, sol[:, 12], label=r'$S_{22}$', color='pink')

ax1.set_xlabel('t (ms)')
ax1.set_ylabel('Mean potentials and synapses')
ax1.grid(True)
ax1.legend(loc='upper left')

# Create a secondary axis for firing rates
ax2 = ax1.twinx()
ax2.plot(t, sol[:, 0], label=r'$r_e$', color='tab:cyan')
ax2.plot(t, sol[:, 4], label=r'$r_i$', color='tab:gray')
ax2.plot(t, sol[:, 8], label=r'$r_2$', color='lightgreen')

ax2.set_ylabel('Firing rates (kHz)')
ax2.legend(loc='upper right')

#plt.title('Mean Potentials, Synapses, and Firing Rates')
plt.show()


# In[13]:


#Define the excitatory input with mu=0 and coherence kappa=2

A = 0.05  # You can change the value of A for the oscillatory input
T = 0.9*Period
mu = 0
k = 2

# External input function 
def p(t):
    I0_value = sp.iv(0, k)  # modified Bessel function of order 0
    return T * (np.exp(k * np.cos(((2 * np.pi) * (t - mu)) / T))) / (T * I0_value)

def p_plot(t):
    I0_value = sp.iv(0, k)  # modified Bessel function of order 0
    # Apply the condition element-wise to the array `t`
    return np.where(t < 200, 0, T * np.exp(k * np.cos((2 * np.pi * (t - mu)) / T)) / (T * I0_value))


# Neural network system of ODEs for the perturbed system. 
#For details on their biological significance, please refer to the thesis.
def neural_network_I_plot(x, t):
    # External inputs for excitatory and inhibitory neurons
    I_e_ext = I_bar_e + tau_e * A * p_plot(t)
    I_i_ext = I_bar_i + tau_i * A * p_plot(t)
    I_2_ext = I_bar_2

    # Compute the total external input to the neurons
    I_e = I_e_ext + tau_e * x[2] - tau_e * x[3]-tau_e*x[10]
    I_i = I_i_ext + tau_i * x[6] - tau_i * x[7]
    I_2 = I_2_ext + tau_2 * x[11] - tau_2 * x[12]

    # System of equations (neuronal dynamics)
    dxdt = [
        (1 / tau_e) * (Delta_e / (np.pi * tau_e) + 2 * x[0] * x[1]),  # re
        (1 / tau_e) * (x[1] ** 2 + eta_e + I_e - (tau_e * np.pi * x[0]) ** 2),  # Ve
        (1 / tau_se) * (-x[2] + Jee * x[0]),  # See
        (1 / tau_si) * (-x[3] + Jei * x[4]),  # Sei
        (1 / tau_i) * (Delta_i / (np.pi * tau_i) + 2 * x[4] * x[5]),  # ri
        (1 / tau_i) * (x[5] ** 2 + eta_i + I_i - (tau_i * np.pi * x[4]) ** 2),  # Vi
        (1 / tau_se) * (-x[6] + Jie * x[0]),  # Sie
        (1 / tau_si) * (-x[7] + Jii * x[4]),   # Sii
        (1 / tau_2) * (Delta_2 / (np.pi * tau_2) + 2 * x[8] * x[9]),  # r2
        (1 / tau_2) * (x[9] ** 2 + eta_2 + I_2 - (tau_2 * np.pi * x[8]) ** 2),  # V2
        (1 / tau_s2) * (-x[10] + Je2 * x[8]),  # Se2
        (1 / tau_se) * (-x[11] + J2e * x[0]),  # S2e
        (1 / tau_s2) * (-x[12] + J22 * x[8])  # S22
    ]
    return dxdt

# Neural network system of ODEs
def neural_network_plot(x, t):
    # External inputs for excitatory and inhibitory neurons
    I_e_ext = I_bar_e + tau_e * A * p_plot(t)
    I_i_ext = I_bar_i + tau_i * A * p_plot(t)

    # Compute the total external input to the neurons
    I_e = I_e_ext + tau_e * x[2] - tau_e * x[3]
    I_i = I_i_ext + tau_i * x[6] - tau_i * x[7]

    # System of equations (neuronal dynamics)
    dxdt = [
        (1 / tau_e) * (Delta_e / (np.pi * tau_e) + 2 * x[0] * x[1]),  # re
        (1 / tau_e) * (x[1] ** 2 + eta_e + I_e - (tau_e * np.pi * x[0]) ** 2),  # Ve
        (1 / tau_se) * (-x[2] + Jee * x[0]),  # See
        (1 / tau_si) * (-x[3] + Jei * x[4]),  # Sei
        (1 / tau_i) * (Delta_i / (np.pi * tau_i) + 2 * x[4] * x[5]),  # ri
        (1 / tau_i) * (x[5] ** 2 + eta_i + I_i - (tau_i * np.pi * x[4]) ** 2),  # Vi
        (1 / tau_se) * (-x[6] + Jie * x[0]),  # Sie
        (1 / tau_si) * (-x[7] + Jii * x[4])   # Sii
    ]
    return dxdt


# Time span for integration
t = np.linspace(0, 750, 2000)

# Solve the system of equations
atol = 1e-17
rtol = 1e-13
sol_plot = odeint(neural_network_plot, x0_david, t, atol=atol, rtol=rtol)
sol_plot_I = odeint(neural_network_I_plot, x0_david_I, t, atol=atol, rtol=rtol)

# Compute the oscillatory term A*p(t)
oscillatory_term = A * p_plot(t)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot r_e and r_i (neural network dynamics)
plt.subplot(2, 1, 1)
plt.plot(t, sol_plot[:, 0], label='$r_e (kHz)$ E-I', color='r')
plt.plot(t, sol_plot_I[:, 0], label='$r_e (kHz)$ E-I-I', color='b')
plt.xlabel('Time (ms)')
plt.ylabel('Activity')
plt.legend()
plt.title('Neural Network Dynamics')
plt.grid(True)

# Plot A*p(t) (oscillatory term)
plt.subplot(2, 1, 2)
plt.plot(t, oscillatory_term, label=r'$A \cdot p(t)$', color='g')
plt.xlabel('Time (ms)')
plt.ylabel(r'$A \cdot p(t)$')
plt.legend()
plt.title('Oscillatory Input $A \cdot p(t)$')
plt.grid(True)

plt.tight_layout()
plt.show()



# In[14]:


# Parameters

A = 0.03  # You can change the value of A for the oscillatory input
T = 0.8*Period
mu = 0
k = 2

# External input function (e.g., sinusoidal)
def p(t,T):
    I0_value = sp.iv(0, k)  # modified Bessel function of order 0
    return T * (np.exp(k * np.cos(((2 * np.pi) * (t - mu)) / T))) / (T * I0_value)

def p_plot(t):
    I0_value = sp.iv(0, k)  # modified Bessel function of order 0
    # Apply the condition element-wise to the array `t`
    return np.where(t < 200, 0, T * np.exp(k * np.cos((2 * np.pi * (t - mu)) / T)) / (T * I0_value))


# Neural network system of ODEs
def neural_network_I_plot(x, t):
    # External inputs for excitatory and inhibitory neurons
    I_e_ext = I_bar_e + tau_e * A * p_plot(t)
    I_i_ext = I_bar_i + tau_i * A * p_plot(t)
    I_2_ext = I_bar_2

    # Compute the total external input to the neurons
    I_e = I_e_ext + tau_e * x[2] - tau_e * x[3]-tau_e*x[10]
    I_i = I_i_ext + tau_i * x[6] - tau_i * x[7]
    I_2 = I_2_ext + tau_2 * x[11] - tau_2 * x[12]

    # System of equations (neuronal dynamics)
    dxdt = [
        (1 / tau_e) * (Delta_e / (np.pi * tau_e) + 2 * x[0] * x[1]),  # re
        (1 / tau_e) * (x[1] ** 2 + eta_e + I_e - (tau_e * np.pi * x[0]) ** 2),  # Ve
        (1 / tau_se) * (-x[2] + Jee * x[0]),  # See
        (1 / tau_si) * (-x[3] + Jei * x[4]),  # Sei
        (1 / tau_i) * (Delta_i / (np.pi * tau_i) + 2 * x[4] * x[5]),  # ri
        (1 / tau_i) * (x[5] ** 2 + eta_i + I_i - (tau_i * np.pi * x[4]) ** 2),  # Vi
        (1 / tau_se) * (-x[6] + Jie * x[0]),  # Sie
        (1 / tau_si) * (-x[7] + Jii * x[4]),   # Sii
        (1 / tau_2) * (Delta_2 / (np.pi * tau_2) + 2 * x[8] * x[9]),  # r2
        (1 / tau_2) * (x[9] ** 2 + eta_2 + I_2 - (tau_2 * np.pi * x[8]) ** 2),  # V2
        (1 / tau_s2) * (-x[10] + Je2 * x[8]),  # Se2
        (1 / tau_se) * (-x[11] + J2e * x[0]),  # S2e
        (1 / tau_s2) * (-x[12] + J22 * x[8])  # S22
    ]
    return dxdt




# Time span for integration
t = np.linspace(0, 2000, 5000)

# Solve the system of equations
atol = 1e-17
rtol = 1e-13
X0=[0,1,1,0,1,1,1,0,1,1,0,1,1]
sol_plot_2 = odeint(neural_network_I_plot, X0, t, atol=atol, rtol=rtol)
sol_plot_I = odeint(neural_network_I_plot, x0_david_I, t, atol=atol, rtol=rtol)

# Compute the oscillatory term A*p(t)
oscillatory_term = A * p_plot(t)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot r_e and r_i (neural network dynamics)
plt.subplot(2, 1, 1)
plt.plot(t, sol_plot_2[:, 0], label='$r_e (kHz)$ E-I', color='r')
plt.plot(t, sol_plot_I[:, 0], label='$r_e (kHz)$ E-I-I', color='b')
plt.xlabel('Time (ms)')
plt.ylabel('Activity')
plt.legend()
plt.title('Neural Network Dynamics')
plt.grid(True)

# Plot A*p(t) (oscillatory term)
plt.subplot(2, 1, 2)
plt.plot(t, oscillatory_term, label=r'$A \cdot p(t)$', color='g')
plt.xlabel('Time (ms)')
plt.ylabel(r'$A \cdot p(t)$')
plt.legend()
plt.title('Oscillatory Input $A \cdot p(t)$')
plt.grid(True)

plt.tight_layout()
plt.show()


# ### Phase equation
# 

# In[15]:


# The following code implements the definitions needed to compute the iPRC.

from scipy.interpolate import interp1d

# Jacobian of the system F(x)
def jacobian(x):
    # Jacobian matrix of the neural network system
    jacobian_matrix = np.zeros((13, 13))

    # For brevity, we'll fill in only the first few rows as an example (you should complete this based on your system).
    jacobian_matrix[0, 0] = (1 / tau_e) * (2 * x[1])  # df0/dx0
    jacobian_matrix[0, 1] = (1 / tau_e) * (2 * x[0])  # df0/dx1
    jacobian_matrix[1, 10] = -1  # df1/dx10
    jacobian_matrix[1, 0] = (-2 * (np.pi**2)*x[0]*tau_e)  # df1/dx0
    jacobian_matrix[1, 1] = (1 / tau_e) * (2 * x[1])  # df1/dx1
    jacobian_matrix[1, 2] = 1  # df1/dx2
    jacobian_matrix[1, 3] = -1  # df1/dx3
    jacobian_matrix[2, 0] = (1 / tau_se) * Jee        # df2/dx0
    jacobian_matrix[2, 2] = (-1 / tau_se)       # df2/dx2
    jacobian_matrix[3, 3] = (-1 / tau_si)   # df3/dx3
    jacobian_matrix[3, 4] = (1 / tau_si) * Jei        # df3/dx4
    jacobian_matrix[4, 4] = (1 / tau_i) * (2 * x[5])  # df4/dx4
    jacobian_matrix[4, 5] = (1 / tau_i) * (2 * x[4])  # df4/dx5
    jacobian_matrix[5, 4] = (-2 * (np.pi**2)*x[4]*tau_i)  # df5/dx4
    jacobian_matrix[5, 5] = (1 / tau_i) * (2 * x[5])  # df5/dx5
    jacobian_matrix[5, 6] = 1  # df5/dx6
    jacobian_matrix[5, 7] = -1  # df5/dx7
    jacobian_matrix[6, 0] = (1 / tau_se) * Jie        # df6/dx0
    jacobian_matrix[6, 6] = (-1 / tau_se)         # df6/dx6
    jacobian_matrix[7, 4] = (1 / tau_si) * Jii        # df7/dx4
    jacobian_matrix[7, 7] = (-1 / tau_si)       # df7/dx7
    jacobian_matrix[8, 8] = (1 / tau_2) * (2 * x[9])  # df8/dx8
    jacobian_matrix[8, 9] = (1 / tau_2) * (2 * x[8])  # df8/dx9
    jacobian_matrix[9, 8] = (-2 * (np.pi**2)*x[8]*tau_2)  # df9/dx8
    jacobian_matrix[9, 9] = (1 / tau_2) * (2 * x[9])  # df9/dx9
    jacobian_matrix[9, 11] = 1  # df9/dx11
    jacobian_matrix[9, 12] = -1  # df9/dx12
    jacobian_matrix[10, 8] = (1 / tau_s2) * Je2        # df10/dx8
    jacobian_matrix[10, 10] = (-1 / tau_s2)       # df10/dx10
    jacobian_matrix[11, 0] = (1 / tau_se) * J2e   # df11/dx0
    jacobian_matrix[11, 11] = -(1 / tau_se)      # df11/dx11
    jacobian_matrix[12, 8] = (1 / tau_s2) * J22   # df12/dx8
    jacobian_matrix[12, 12] = (-1 / tau_s2)        # df12/dx12


    return -jacobian_matrix.T


# Define the matrix differential equation as a vector-valued function
def adjoint_equation(X_flat, t, interp_orbit):
    # Reshape the flat vector back into the matrix form (8x8 in this case)
    X = X_flat.reshape((13, 13))  # Adjust the shape for your matrix size

    # Interpolate the state x(t) from the periodic orbit at time t
    x_t = interp_orbit(t)  # Interpolate each x component

    # Compute the Jacobian at x(t)
    J_t = jacobian(x_t)

    # Compute the derivative of the matrix dX/dt = A(t) * X
    dX_dt = J_t @ X

    # Flatten the result back into a vector
    return dX_dt.flatten()


# Define the vectorial differential equation as a vector-valued function
def vector_adjoint_equation(x, t, interp_orbit):
    x_t = interp_orbit(t)  # Interpolate each x component

    # Compute the Jacobian at x(t)
    J_t = jacobian(x_t)

    # Compute the derivative of the matrix dX/dt = A(t) * X
    dX_dt = J_t @ x

    # Flatten the result back into a vector
    return dX_dt

#INTEGRATION OF THE PHASE EQUATION
def phase_equation(x, t, interp_orbit_iPRC,A,T):
    z_t = interp_orbit_iPRC(x % Period)  # Interpolate z_t theta

    # Compute the phase equation (wrapped modulo Period)
    dtheta = 1 +  (z_t[1] + z_t[5]) * A * p(t,T)
    return dtheta



# In[18]:


#ALL THIS COMPUTATITIONS ARE INTRINSIC OF THE OSCILLATOR (do not depend on theta,T,A)
# The following code implements the computation of the iPRC.


x_0 = P_1  # Initial condition for the periodic orbit (assumed to be defined)

t = np.linspace(0, Period, 1000)

# Integrate the periodic orbit
orbit_solution = odeint(neural_network_notperturbed, x_0, t, atol=atol, rtol=rtol)
interp_orbit_oscillator = interp1d(t, orbit_solution.T, kind='cubic', fill_value='extrapolate')

# Initial condition for the fundamental matrix (identity matrix)
A0 = np.eye(13).flatten()
t_vals = np.linspace(0, Period, 1000)

# Integrate the adjoint equation for the fundamental matrix
sol = odeint(adjoint_equation, A0, t_vals, args=(interp_orbit_oscillator,), atol=atol, rtol=rtol)
A_T = sol[-1].reshape(13, 13)
eigenvalues, eigenvectors = np.linalg.eig(A_T)
print(eigenvalues)
real_eigenvalues = np.real(eigenvalues)
closest_index = np.argmin(np.abs(real_eigenvalues - 1))
print(closest_index)
v_0 = np.real(eigenvectors[:, closest_index])
alpha = 1 / (v_0 @ neural_network_notperturbed(x_0, 0))
V_0 = alpha * v_0

t_vals = np.linspace(0, Period, 1000)
sol = odeint(adjoint_equation, A0, t_vals, args=(interp_orbit_oscillator,), atol=atol, rtol=rtol)
# Integrate the adjoint equation for the phase response curve
sol_phi_t = odeint(vector_adjoint_equation, V_0, t_vals, args=(interp_orbit_oscillator,), atol=atol, rtol=rtol)
interp_orbit_iPRC = interp1d(t_vals, sol_phi_t.T, kind='cubic', fill_value='extrapolate')
# Reshape sol into 3D array (len(t_vals), 13, 13)
sol_reshaped = sol.reshape(len(t_vals), 13, 13)

# Multiply each reshaped matrix by V_0
INTERP_IPRC = np.array([A @ V_0 for A in sol_reshaped])

#plt.plot(t_vals[0:6000],sol_phi_t[0:6000,1],label=r'$iPRC-V_e$')
#plt.plot(t_vals[0:6000],sol_phi_t[0:6000,5],label=r'$iPRC-V_i$')
#plt.plot(t_vals,sol_phi_t[0:500,1]+sol_phi_t[0:500,5],label=r'$iPRC-V_i$')
plt.plot(t_vals,sol_phi_t[:,1],label=r'$iPRC-V_e$')
plt.plot(t_vals,sol_phi_t[:,5],label=r'$iPRC-V_i$')


# In[19]:


# Define the stroboscopic map function with updated parameters (x, A, T)
def stroboscopic_map(x, A, T,interp_orbit_iPRC):

    # Integrate the phase equation with the initial condition x
    t_phase = np.linspace(0, T, 1000)
    sol_theta_t = odeint(phase_equation, x, t_phase, args=(interp_orbit_iPRC, A, T), atol=atol, rtol=rtol)
    Theta_n = sol_theta_t[-1] % Period

    return Theta_n[0] #to ensure that it is an scalar



theta_n=18
#print(T/Period)
for i in range(1,50):
  theta_n=stroboscopic_map(theta_n,0.05,0.95*Period, interp_orbit_iPRC)
  print(theta_n)


# In[25]:


# Let us define a function to generate a COBWEB plot
def cobweb_plot(x0, T, A, num_steps):
    # Prepare the figure
    plt.figure(figsize=(6, 6))

    # Define the range for x values
    x= np.linspace(0, Period, 150)
    #x1 = np.linspace(0, 18.1, 100)
    #x2=np.linspace(18.4,Period,100)
    y = [stroboscopic_map(val, A, T, interp_orbit_iPRC) for val in x]  # Map values for y-axis
    #y2 = [stroboscopic_map(val, A, T, interp_orbit_iPRC) for val in x2]  # Map values for y-axis

    # Plot y = f(x) and y = x
    plt.plot(x, y, label='Stroboscopic Map', color='pink')
    #plt.plot(x2, y2, color='pink')
    plt.plot(x, x, label='y = x', color='blue')

    # Plot the cobweb diagram
    x_n = x0
    for _ in range(num_steps):
        x_next = stroboscopic_map(x_n, A, T, interp_orbit_iPRC)  # compute the following point

        # Vertical line (x_n, f(x_n))
        plt.plot([x_n, x_n], [x_n, x_next], color='black', lw=1)

        # Horizontal line (f(x_n), x_{n+1})
        plt.plot([x_n, x_next], [x_next, x_next], color='black', lw=1)

        # Update to the next point
        x_n = x_next

    # Add plot details
    plt.title(f'Cobweb Diagram for A={A}, T={T}')
    plt.xlabel(r'$\theta_n$')
    plt.ylabel(r'$\theta_{n+1} = P(\theta_n)$')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example call to the function with initial parameters
cobweb_plot(0, 1.5*Period, 0.04, 10)


# ### Computation of Arnold Tongues for $\kappa=\infty$

# In[19]:


# The following code calculates the 1:1 and 2:1 Arnold tongues for pulsatile inputs.

time_points = np.linspace(0, Period, 1000)  # Sample points across one period
interp_values = interp_orbit_iPRC(time_points)  # Evaluate interp_orbit_iPRC over these points

# Extract components 1 and 5, and calculate their sum
Z_values = interp_values[1] + interp_values[5]  # Sum of component 1 and component 5

# Calculate Zmax and Zmin
Zmax = np.max(Z_values)
Zmin = np.min(Z_values)
print(Zmax)
print(Zmin)

# Define the range for A and calculate corresponding T/T* values
A_values = np.linspace(0, 0.2, 500)
T_over_Tstar_max = 1 + A_values * Zmax
T_over_Tstar_min = 1 + A_values * Zmin

T_over_Tstar_max_2 = (1/2)*(1 + A_values * Zmax)
T_over_Tstar_min_2 = (1/2)*(1 + A_values * Zmin)


T_over_Tstar_max = T_over_Tstar_max[T_over_Tstar_max > 0]
T_over_Tstar_min = T_over_Tstar_min[T_over_Tstar_min > 0]

T_over_Tstar_max_2 = T_over_Tstar_max_2[T_over_Tstar_max_2 > 0]
T_over_Tstar_min_2 = T_over_Tstar_min_2[T_over_Tstar_min_2 > 0]

# Plotting
plt.figure(figsize=(8, 6))
fill = np.full(len(A_values), 10)


plt.fill_betweenx(A_values[:len(T_over_Tstar_min)], 1 / T_over_Tstar_min[:len(T_over_Tstar_min)], 1 / T_over_Tstar_max[:len(T_over_Tstar_min)], color='lightblue', alpha=0.5, label='Region (blue)')
plt.fill_betweenx(A_values[len(T_over_Tstar_min)+1:], 1 / T_over_Tstar_max[len(T_over_Tstar_min)+1:],fill[len(T_over_Tstar_min)+1:], color='lightblue', alpha=0.5, label='Region (blue)')
# Plot the boundary lines for the filled region
plt.plot(1/T_over_Tstar_max, A_values[:len(T_over_Tstar_max)], color='lightblue')
plt.plot(1/T_over_Tstar_min, A_values[:len(T_over_Tstar_min)], color='lightblue')


plt.fill_betweenx(A_values[:len(T_over_Tstar_min_2)], 1 / T_over_Tstar_min_2[:len(T_over_Tstar_min_2)], 1 / T_over_Tstar_max_2[:len(T_over_Tstar_min_2)], color='purple', alpha=0.5)
plt.fill_betweenx(A_values[len(T_over_Tstar_min_2)+1:], 1 / T_over_Tstar_max_2[len(T_over_Tstar_min_2)+1:],fill[len(T_over_Tstar_min_2)+1:], color='purple', alpha=0.5)


# Plot the boundary lines for the filled region
plt.plot(1/T_over_Tstar_max_2, A_values[:len(T_over_Tstar_max_2)], color='purple')
plt.plot(1/T_over_Tstar_min_2, A_values[:len(T_over_Tstar_min_2)], color='purple')



# Setting limits for the axes
plt.xlim(0, 5.5)
plt.ylim(0, 0.2)

# Labels and title
plt.xlabel(r'$\frac{T}{T^*}$')
plt.ylabel("A")

plt.grid(True)

plt.show()


# In[ ]:


# The following code implements the stroboscopic map for pulsatile inputs. 
#It will help us numerically calculate the 1:2 Arnold tongue as kappa tends to infinity.

def poincare_map(x, A, T,interp_orbit_iPRC):
    val=interp_orbit_iPRC(x)
    val=val[1]+val[5]
    return (x+T+A*T*val)%Period

def poincare_map_2(x,A,T,interp_orbit_iPRC):
    x1=poincare_map(x, A, T,interp_orbit_iPRC)
    x2=poincare_map(x1, A, T,interp_orbit_iPRC)
    return x2


# In[ ]:


# Let us define a function to generate a COBWEB plot
def cobweb_plot_squared(x0, T, A, num_steps):
    # Prepare the figure
    plt.figure(figsize=(6, 6))

    # Define the range for x values
    x = np.linspace(0, Period, 100)
    y = [poincare_map_2(val,A,T, interp_orbit_iPRC) for val in x]  # Map values for y-axis
   

    # Plot y = f(x) and y = x
    plt.plot(x, y, label='Stroboscopic 5 Map', color='pink')
    
    plt.plot(x, x, label='y = x', color='blue')

     # Plot the cobweb diagram
    x_n = x0
    for _ in range(num_steps):
        x_next = poincare_map_2(x_n,A,T, interp_orbit_iPRC)  # compute the following point

        # Vertical line (x_n, f(x_n))
        plt.plot([x_n, x_n], [x_n, x_next], color='black', lw=1)

        # Horizontal line (f(x_n), x_{n+1})
        plt.plot([x_n, x_next], [x_next, x_next], color='black', lw=1)

        # Update to the next point
        x_n = x_next

    # Add plot details
    plt.title(f'Cobweb Diagram for A={A}, T={T}')
    plt.xlabel(r'$\theta_n$')
    plt.ylabel(r'$\theta_{n+1} = P(\theta_n)$')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example call to the function with initial parameters
cobweb_plot_squared(0, 0.5913*Period, 0.025, 100)


# In[ ]:


# The following code helps us find a point on the boundary of the Arnold tongue.

import numpy as np
from scipy.optimize import fsolve
import numdifftools as nd

def derivative_stroboscopic_map_analytic(theta, A, T, interp_orbit_iPRC):
    stroboscopic_map_theta = lambda theta: poincare_map_2(theta, A, T, interp_orbit_iPRC)
    dPhi_dtheta = nd.Derivative(stroboscopic_map_theta)(theta)
    return dPhi_dtheta

# Set T to be 95% of the period
T = 0.5913*Period

# Initial guess for [theta, A]
initial_guess = [22, 0.025]

# Define the function to find roots for [theta, A] given a fixed T
def phase_conditions_root(vars):
    theta, A = vars  # Unpack variables

   
    Phi_T = poincare_map_2(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    
    dPhi_dtheta = derivative_stroboscopic_map_analytic(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Use fsolve to find a root, starting from the initial guess
solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

# Extract the result
theta_solution, A_solution = solution.x

# Check conditions by evaluating phase_conditions_root at the solution
cond1, cond2 = phase_conditions_root([theta_solution, A_solution])

# Output the solution and the values of the conditions
print("Solution found:")
print(f"theta = {theta_solution}")
print(f"A = {A_solution}")
print("\nVerification of conditions:")
print(f"Condition 1 (Phi_T - theta): {cond1}")
print(f"Condition 2 (dPhi/dT - 1): {cond2}")

# Check if both conditions are close to zero
tolerance = 1e-6
if abs(cond1) < tolerance and abs(cond2) < tolerance:
    print("\nBoth conditions are satisfied within the specified tolerance.")
else:
    print("\nWarning: One or both conditions are not satisfied within the specified tolerance.")


# In[43]:


# The following code calculates the 1:2 Arnold tongue for pulsatile inputs. 
#The explanation of the method can be found in the appendix of the written thesis.
num_iterations_up_r = 20  # Number of iterations you want

# Initialize lists to store results
T_values_right = []
A_values_right = []

# Initialize T to 95% of Period
A = 0.025

# Initial conditions
initial_guess = [19.79, 0.4725*Period]

# Function to find roots
def phase_conditions_root(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = poincare_map_2(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map_analytic(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    #print(T/Period)
    # Prepare the new T
    A += 0.01  # Increment A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 5
A = 0.025
# Initial conditions
initial_guess = [19.79, 0.4725*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    #print(T/Period)
    # Prepare the new T
    A -= 0.01  # Decrement A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Set the increment h

num_iterations_up_left = 50  # Number of iterations you want
def phase_conditions_root(vars):
    theta, A = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = poincare_map_2(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map_analytic(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])
# Initialize lists to store results
T_values = []
A_values = []

# Initialize T to 95% of Period
T = 0.5913*Period

# Initial conditions
initial_guess = [23.73, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_up_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values.append(T / Period)  # Store normalized T
    A_values.append(A_solution)    # Store A

    #print(T/Period)
    # Prepare the new T
    T += 0.05*Period  # Increment T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point

num_iterations_down_left = 10
T = 0.5913*Period
# Initial conditions
initial_guess = [23.73, 0.025]


# Loop to iterate and find new points
for i in range(num_iterations_down_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values.append(T / Period)  # Store normalized T
    A_values.append(A_solution)    # Store A

    #print(T/Period)
    # Prepare the new T
    T -= 0.01*Period  # Decrement T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), for example, from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate both curves for these common points in terms of A
interp_left = interp1d(A_values, T_values, bounds_error=False, fill_value="extrapolate")
interp_right = interp1d(A_values_right, T_values_right, bounds_error=False, fill_value="extrapolate")

# Evaluate both interpolations at the common A points
T_values_left_interp = interp_left(A_common)
T_values_right_interp = interp_right(A_common)

# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right[0:num_iterations_up_r], A_values_right[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right[num_iterations_up_r:], A_values_right[num_iterations_up_r:], color='blue')
plt.plot(T_values[0:num_iterations_up_left], A_values[0:num_iterations_up_left], color='blue')
plt.plot(T_values[num_iterations_up_left:], A_values[num_iterations_up_left:], color='blue')

# Fill the area between the two interpolated curves
plt.fill_betweenx(A_common, T_values_left_interp, T_values_right_interp, color='lightblue', alpha=0.5)

# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()

# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_left_interp_eii_1_2_inf.npy", T_values_left_interp)
np.save("T_values_right_interp_eii_1_2_inf.npy", T_values_right_interp)


# ### Computation of Arnold tongues for $\kappa \neq \infty$

# In[22]:


import numpy as np
from scipy.optimize import fsolve
# Suponiendo que T y A están definidos
# Ejemplo: T = 1.0, A = 0.05 (ajusta a los valores deseados)
T = 2.81*Period
A = 0.025

stroboscopic_map_theta = lambda theta: stroboscopic_map(theta, A, T, interp_orbit_iPRC)


def fixed_point_condition(theta):
    return stroboscopic_map_theta(theta) - theta


initial_theta = 10  

theta_solution = fsolve(fixed_point_condition, initial_theta)[0]

print("Valor de theta encontrado:", theta_solution)

verificacion = fixed_point_condition(theta_solution)
print("Verificación (debe ser cercano a 0):", verificacion)


# In[25]:


import numpy as np
from scipy.optimize import fsolve

def derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC):
    stroboscopic_map_theta = lambda theta: stroboscopic_map(theta, A, T, interp_orbit_iPRC)
    dPhi_dtheta = nd.Derivative(stroboscopic_map_theta)(theta)
    return dPhi_dtheta

# Set T to be 95% of the period
A=0.01
# Initial guess for [theta, T]
initial_guess = [28, 1.99*Period]

# Define the function to find roots for [theta, A] given a fixed T
def phase_conditions_root_A(vars):
    theta, T = vars  # Unpack variables

    
    Phi_T = stroboscopic_map(theta,A,T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    
    dPhi_dtheta = derivative_stroboscopic_map(theta,A,T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Use fsolve to find a root, starting from the initial guess
solution = root(phase_conditions_root_A, initial_guess, method='hybr', tol=1e-8)

# Extract the result
theta_solution, T_solution = solution.x

# Check conditions by evaluating phase_conditions_root at the solution
cond1, cond2 = phase_conditions_root_A([theta_solution, T_solution])

# Output the solution and the values of the conditions
print("Solution found:")
print(f"theta = {theta_solution}")
print(f"T/T^* = {T_solution/Period}")
print("\nVerification of conditions:")
print(f"Condition 1 (Phi_T - theta): {cond1}")
print(f"Condition 2 (dPhi/dT - 1): {cond2}")

# Check if both conditions are close to zero
tolerance = 1e-6
if abs(cond1) < tolerance and abs(cond2) < tolerance:
    print("\nBoth conditions are satisfied within the specified tolerance.")
else:
    print("\nWarning: One or both conditions are not satisfied within the specified tolerance.")


# In[252]:


import numpy as np
from scipy.optimize import fsolve

def derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC):
    stroboscopic_map_theta = lambda theta: stroboscopic_map(theta, A, T, interp_orbit_iPRC)
    dPhi_dtheta = nd.Derivative(stroboscopic_map_theta)(theta)
    return dPhi_dtheta

# Set T to be 95% of the period
T=1.45*Period
# Initial guess for [theta, A]
initial_guess = [11.29,0.025]

# Define the function to find roots for [theta, A] given a fixed T

def phase_conditions_root(vars):
    theta, A = vars  # Unpack variables

    
    Phi_T = stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

   
    dPhi_dtheta = derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])
# Use fsolve to find a root, starting from the initial guess
solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

# Extract the result
theta_solution, A_solution = solution.x

# Check conditions by evaluating phase_conditions_root at the solution
cond1, cond2 = phase_conditions_root([theta_solution, A_solution])

# Output the solution and the values of the conditions
print("Solution found:")
print(f"theta = {theta_solution}")
print(f"A = {A_solution}")
print("\nVerification of conditions:")
print(f"Condition 1 (Phi_T - theta): {cond1}")
print(f"Condition 2 (dPhi/dT - 1): {cond2}")

# Check if both conditions are close to zero
tolerance = 1e-6
if abs(cond1) < tolerance and abs(cond2) < tolerance:
    print("\nBoth conditions are satisfied within the specified tolerance.")
else:
    print("\nWarning: One or both conditions are not satisfied within the specified tolerance.")


# In[ ]:


#COMPUTATION OF THE 1:1 ARNOLD TONGUE
# Set the step increment h
h = 0.001  # The step size you desire
num_iterations_up_r = 60  # Number of iterations you want

# Initialize lists to store results
T_values_right = []
A_values_right = []

# Initialize T to 95% of Period
A = 0.05

# Initial conditions
initial_guess = [28.9206, 0.857*Period]

# Function to find roots
def phase_conditions_root(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    # Prepare the new T
    A -= 0.001  # Decrease A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 155
A = 0.05

# Initial conditions
initial_guess = [28.9206, 0.857*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    # Prepare the new T
    A += 0.001  # Increase A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Set the step increment h
h = 0.01  # The step size you desire
num_iterations_up_left = 110  # Number of iterations you want

# Initialize lists to store results
T_values = []
A_values = []

# Initialize T to 95% of Period
T = 1.45 * Period

# Initial conditions
initial_guess = [11.29, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_up_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values.append(T / Period)  # Store normalized T
    A_values.append(A_solution)    # Store A

    # Prepare the new T
    T += h * Period  # Increase T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point

num_iterations_down_left = 25
T = 1.45 * Period
# Initial conditions
initial_guess = [11.29, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_down_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values.append(T / Period)  # Store normalized T
    A_values.append(A_solution)    # Store A

    # Prepare the new T
    T -= h * Period  # Decrease T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), for example, from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate both curves for these common points in terms of A
interp_left = interp1d(A_values, T_values, bounds_error=False, fill_value="extrapolate")
interp_right = interp1d(A_values_right, T_values_right, bounds_error=False, fill_value="extrapolate")

# Evaluate both interpolations at the common A points
T_values_left_interp = interp_left(A_common)
T_values_right_interp = interp_right(A_common)

# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right[0:num_iterations_up_r], A_values_right[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right[num_iterations_up_r:], A_values_right[num_iterations_up_r:], color='blue')
plt.plot(T_values[0:num_iterations_up_left], A_values[0:num_iterations_up_left], color='blue')
plt.plot(T_values[num_iterations_up_left:], A_values[num_iterations_up_left:], color='blue')

# Fill the area between the two interpolated curves
plt.fill_betweenx(A_common, T_values_left_interp, T_values_right_interp, color='lightblue', alpha=0.5)

# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()

# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_left_interp_I.npy", T_values_left_interp)
np.save("T_values_right_interp_I.npy", T_values_right_interp)


# In[91]:


# Set the step increment h
h = 0.001  # The step size you desire
num_iterations_up_r = 50  # Number of iterations you want

# Initialize lists to store results
T_values_right = []
A_values_right = []

# Initialize T to 95% of Period
A = 0.05

# Initial conditions
initial_guess = [28.9206, 0.857*Period]

# Function to find roots
def phase_conditions_root(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    print(A)
    # Prepare the new T
    A -= 0.001  # Decrease A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 151
A = 0.05

# Initial conditions
initial_guess = [28.9206, 0.857*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right.append(T_solution / Period)  # Store normalized T
    A_values_right.append(A)    # Store A

    print(A)
    # Prepare the new T
    A += 0.001  # Increase A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), for example, from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate the right curve for these common points in terms of A
interp_right = interp1d(A_values_right, T_values_right, bounds_error=False, fill_value="extrapolate")

# Evaluate the interpolation at the common A points
T_values_right_interp = interp_right(A_common)

# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right[0:num_iterations_up_r], A_values_right[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right[num_iterations_up_r:], A_values_right[num_iterations_up_r:], color='blue')

# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()

# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_right_interp_I.npy", T_values_right_interp)


# In[63]:


#COMPUTATION OF THE 2:1 ARNOLD TONGUE
# Set the step increment h
h = 0.001  # The step size you desire
num_iterations_up_r = 20  # Number of iterations you want

# Initialize lists to store results
T_values_right_2 = []
A_values_right_2 = []

# Initialize T to 95% of Period
A = 0.01

# Initial conditions
initial_guess = [29.54, 1.99*Period]

# Function to find roots
def phase_conditions_root(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right_2.append(T_solution / Period)  # Store normalized T
    A_values_right_2.append(A)    # Store A
    
    print(A)
    # Prepare the new T
    A -= 0.0005  # Decrease A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 130
A = 0.01

# Initial conditions
initial_guess = [29.54, 1.99*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store values in the lists
    T_values_right_2.append(T_solution / Period)  # Store normalized T
    A_values_right_2.append(A)    # Store A

    print(A)
    # Prepare the new T
    A += 0.0005  # Increase A
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), for example, from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate the right curve for these common points in terms of A
interp_right = interp1d(A_values_right_2, T_values_right_2, bounds_error=False, fill_value="extrapolate")

# Evaluate the interpolation at the common A points
T_values_right_interp_2_I = interp_right(A_common)

# Function to find roots for A
def phase_conditions_root_A(vars):
    theta, A = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])


# Set the step increment h
h = 0.02  # The step size you desire
num_iterations_up_left = 40  # Number of iterations you want

# Initialize lists to store results
T_values_2_I = []
A_values_2_I = []

T = 2.81 * Period
# Initial conditions
initial_guess = [12.31, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_up_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_A, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values_2_I.append(T / Period)  # Store normalized T
    A_values_2_I.append(A_solution)    # Store A

    print(T / Period)
    # Prepare the new T
    T += h * Period  # Increase T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point

num_iterations_down_left = 42
T = 2.81 * Period
# Initial conditions
initial_guess = [12.31, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_down_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_A, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store values in the lists
    T_values_2_I.append(T / Period)  # Store normalized T
    A_values_2_I.append(A_solution)    # Store A

    print(T / Period)
    # Prepare the new T
    T -= h * Period  # Decrease T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), for example, from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate both curves for these common points in terms of A
interp_left = interp1d(A_values_2_I, T_values_2_I, bounds_error=False, fill_value="extrapolate")

# Evaluate both interpolations at the common A points
T_values_left_interp_2_I = interp_left(A_common)

# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_left_interp_2_I.npy", T_values_left_interp_2_I)

# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right_2[0:num_iterations_up_r], A_values_right_2[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right_2[num_iterations_up_r:], A_values_right_2[num_iterations_up_r:], color='blue')
plt.plot(T_values_2_I[0:num_iterations_up_left], A_values_2_I[0:num_iterations_up_left], color='blue')
plt.plot(T_values_2_I[num_iterations_up_left:], A_values_2_I[num_iterations_up_left:], color='blue')

# Fill the area between the two interpolated curves
plt.fill_betweenx(A_common, T_values_left_interp_2_I, T_values_right_interp_2_I, color='lightblue', alpha=0.5)

# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()

# Saving the right interpolated data to .npy file
np.save("T_values_right_interp_2_I.npy", T_values_right_interp_2_I)




# In[42]:


# Define the function for P^2(theta) in order to find the 1:2 Arnold tongue
def stroboscopic_map_squared(x, A, T,interp_orbit_iPRC):
    # Apply stroboscopic_map twice
    first_application = stroboscopic_map(x, A, T, interp_orbit_iPRC)
    second_application = stroboscopic_map(first_application, A, T, interp_orbit_iPRC)
    return second_application


# In[44]:


# Example call to the function with initial parameters
def cobweb_plot(x0, T, A, num_steps):
    # Prepare the figure
    plt.figure(figsize=(6, 6))

    # Define the range for x values
    x = np.linspace(0, Period, 100)
    y = [stroboscopic_map(val, A, T, interp_orbit_iPRC) for val in x]  # Map values for y-axis

    # Plot y = f(x) and y = x
    plt.plot(x, y, label='Stroboscopic Map', color='pink')
    plt.plot(x, x, label='y = x', color='blue')

    # Plot the cobweb diagram
    x_n = x0
    for _ in range(num_steps):
        x_next = stroboscopic_map(x_n, A, T, interp_orbit_iPRC)  # compute the following point

        # Vertical line (x_n, f(x_n))
        plt.plot([x_n, x_n], [x_n, x_next], color='black', lw=1)

        # Horizontal line (f(x_n), x_{n+1})
        plt.plot([x_n, x_next], [x_next, x_next], color='black', lw=1)

        # Update to the next point
        x_n = x_next

    # Add plot details
    plt.title(f'Cobweb Diagram for A={A}, T={T}')
    plt.xlabel(r'$\theta_n$')
    plt.ylabel(r'$\theta_{n+1} = P(\theta_n)$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def cobweb_plot_squared(x0, T, A, num_steps):
    # Prepare the figure
    plt.figure(figsize=(6, 6))

    # Define the range for x values
    x = np.linspace(0, Period, 100)
    y = [stroboscopic_map_squared(val, A, T, interp_orbit_iPRC) for val in x]  # Map values for y-axis

    # Plot y = f(x) and y = x
    plt.plot(x, y, label='Stroboscopic Map', color='pink')
    plt.plot(x, x, label='y = x', color='blue')

    # Plot the cobweb diagram
    x_n = x0
    for _ in range(num_steps):
        x_next = stroboscopic_map_squared(x_n, A, T, interp_orbit_iPRC)  # compute the following point

        # Vertical line (x_n, f(x_n))
        plt.plot([x_n, x_n], [x_n, x_next], color='black', lw=1)

        # Horizontal line (f(x_n), x_{n+1})
        plt.plot([x_n, x_next], [x_next, x_next], color='black', lw=1)

        # Update to the next point
        x_n = x_next
        
        
    x = np.linspace(0, Period, 100)
    y = [stroboscopic_map(val, A, T, interp_orbit_iPRC) for val in x]  # Map values for y-axis

    # Plot y = f(x) and y = x
    plt.plot(x, y, label='Stroboscopic Map', color='red')
    plt.plot(x, x, label='y = x', color='blue')

    # Plot the cobweb diagram
    x_n = x0
    for _ in range(num_steps):
        x_next = stroboscopic_map(x_n, A, T, interp_orbit_iPRC)  # compute the following point

        # Vertical line (x_n, f(x_n))
        plt.plot([x_n, x_n], [x_n, x_next], color='black', lw=1)

        # Horizontal line (f(x_n), x_{n+1})
        plt.plot([x_n, x_next], [x_next, x_next], color='black', lw=1)

        # Update to the next point
        x_n = x_next
    # Add plot details
    plt.title(f'Cobweb Diagram for A={A}, T={T}')
    plt.xlabel(r'$\theta_n$')
    plt.ylabel(r'$\theta_{n+1} = P(\theta_n)$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
#cobweb_plot(0, Period, 0.05, 50)
cobweb_plot_squared(0, 2*Period, 0.075, 10)


# In[66]:


import numpy as np
from scipy.optimize import fsolve

def derivative_stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC):
    stroboscopic_map_theta_squared = lambda theta: stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)
    dPhi_dtheta = nd.Derivative(stroboscopic_map_theta_squared)(theta)
    return dPhi_dtheta

# Set T to be 95% of the period
T = 0.56*Period

# Initial guess for [theta, A]
initial_guess = [10, 0.025]

# Define the function to find roots for [theta, A] given a fixed T
def phase_conditions_root_squared(vars):
    theta, A = vars  # Unpack variables

    
    Phi_T = stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    
    dPhi_dtheta = derivative_stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Use fsolve to find a root, starting from the initial guess
solution = root(phase_conditions_root_squared, initial_guess, method='hybr', tol=1e-8)

# Extract the result
theta_solution, A_solution = solution.x

# Check conditions by evaluating phase_conditions_root at the solution
cond1, cond2 = phase_conditions_root_squared([theta_solution, A_solution])

# Output the solution and the values of the conditions
print("Solution found:")
print(f"theta = {theta_solution}")
print(f"A = {A_solution}")
print("\nVerification of conditions:")
print(f"Condition 1 (Phi_T - theta): {cond1}")
print(f"Condition 2 (dPhi/dT - 1): {cond2}")

# Check if both conditions are close to zero
tolerance = 1e-6
if abs(cond1) < tolerance and abs(cond2) < tolerance:
    print("\nBoth conditions are satisfied within the specified tolerance.")
else:
    print("\nWarning: One or both conditions are not satisfied within the specified tolerance.")


# In[68]:


#COMPUTATION OF THE 1:2 ARNOLD TONGUE
# Set the step increment h
h = 0.001  # The desired step size
num_iterations_up_r = 25  # Number of iterations you want

# Initialize lists to store results
T_values_right_1 = []
A_values_right_1 = []

# Initialize T to 95% of Period
A = 0.025

# Initial conditions
initial_guess = [17.74, 0.52*Period]

# Function to find roots
def phase_conditions_root_squared(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store the results in the lists
    T_values_right_1.append(T_solution / Period)  # Store normalized T
    A_values_right_1.append(A)    # Store A

    print(A)
    # Prepare the new T
    A -= 0.001  # Decrease T
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 175
A = 0.025

# Initial conditions
initial_guess = [17.74, 0.52*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store the results in the lists
    T_values_right_1.append(T_solution / Period)  # Store normalized T
    A_values_right_1.append(A)    # Store A

    print(A)
    # Prepare the new T
    A += 0.001  # Decrease T
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), e.g., from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

interp_right = interp1d(A_values_right_1, T_values_right_1, bounds_error=False, fill_value="extrapolate")

T_values_right_interp_1_I = interp_right(A_common)


# Function to find roots
def phase_conditions_root_squared_A(vars):
    theta, A = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])


# Set the step increment h
h = 0.02  # The desired step size
num_iterations_up_left = 70  # Number of iterations you want

# Initialize lists to store results
T_values_1_I = []
A_values_1_I = []

T = 0.56 * Period
# Initial conditions
initial_guess = [10.317, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_up_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared_A, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store the results in the lists
    T_values_1_I.append(T / Period)  # Store normalized T
    A_values_1_I.append(A_solution)    # Store A

    print(T / Period)
    # Prepare the new T
    T += h * Period  # Increase T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point

num_iterations_down_left = 15
T = 0.56 * Period
# Initial conditions
initial_guess = [10.317, 0.025]

# Loop to iterate and find new points
for i in range(num_iterations_down_left):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared_A, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, A_solution = solution.x

    # Store the results in the lists
    T_values_1_I.append(T / Period)  # Store normalized T
    A_values_1_I.append(A_solution)    # Store A

    print(T / Period)
    # Prepare the new T
    T -= h * Period  # Decrease T
    initial_guess = [theta_solution, A_solution]  # Use the found solution as the new initial point


# Create a common set of points for the y-axis (A), e.g., from 0 to 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolate both curves for these common points based on A
interp_left = interp1d(A_values_1_I, T_values_1_I, bounds_error=False, fill_value="extrapolate")

# Evaluate both interpolations at the common A points
T_values_left_interp_1_I = interp_left(A_common)


plt.plot(T_values_1_I[0:num_iterations_up_left], A_values_1_I[0:num_iterations_up_left], color='blue')
plt.plot(T_values_1_I[num_iterations_up_left:], A_values_1_I[num_iterations_up_left:], color='blue')

# Fill the area between the two interpolated curves
plt.fill_betweenx(A_common, T_values_left_interp_1_I, T_values_right_interp_1_I, color='lightblue', alpha=0.5)


# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()


# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_left_interp_1_I.npy", T_values_left_interp_1_I)


# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right_1[0:num_iterations_up_r], A_values_right_1[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right_1[num_iterations_up_r:], A_values_right_1[num_iterations_up_r:], color='blue')


# Labels and additional settings
plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()


np.save("T_values_right_interp_1_I.npy", T_values_right_interp_1_I)


# In[85]:


# Set the step increment h
h = 0.001  # The desired step size
num_iterations_up_r = 25  # Number of iterations you want

# Initialize lists to store results
T_values_right_1_p = []
A_values_right_1_p = []

# Initialize T to 95% of Period
A = 0.025

# Initial conditions
initial_guess = [10.317, 0.56*Period]

# Function to find roots
def phase_conditions_root_squared(vars):
    theta, T = vars  # Unpack variables

    # Evaluate the stroboscopic map
    Phi_T = stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 1: Phi_T - theta = 0
    cond1 = Phi_T - theta

    # Derivative with respect to theta
    dPhi_dtheta = derivative_stroboscopic_map_squared(theta, A, T, interp_orbit_iPRC)

    # Condition 2: dPhi/dT - 1 = 0
    cond2 = dPhi_dtheta - 1

    return np.array([cond1, cond2])

# Loop to iterate and find new points
for i in range(num_iterations_up_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store the results in the lists
    T_values_right_1_p.append(T_solution / Period)  # Store normalized T
    A_values_right_1_p.append(A)    # Store A

    print(A)
    # Prepare the new T
    A -= 0.001  # Decrease T
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point

num_iterations_down_r = 175
A = 0.025

# Initial conditions
initial_guess = [10.317, 0.56*Period]

# Loop to iterate and find new points
for i in range(num_iterations_down_r):
    # Use fsolve to find a root
    solution = root(phase_conditions_root_squared, initial_guess, method='hybr', tol=1e-8)

    # Extract results
    theta_solution, T_solution = solution.x

    # Store the results in the lists
    T_values_right_1_p.append(T_solution / Period)  # Store normalized T
    A_values_right_1_p.append(A)    # Store A

    print(A)
    # Prepare the new T
    A += 0.001  # Increase T
    initial_guess = [theta_solution, T_solution]  # Use the found solution as the new initial point


# Plot the original curves
plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right_1_p[0:num_iterations_up_r], A_values_right_1_p[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right_1_p[num_iterations_up_r:], A_values_right_1_p[num_iterations_up_r:], color='blue')


# In[87]:


interp_left = interp1d(A_values_right_1_p, T_values_right_1_p, bounds_error=False, fill_value="extrapolate")

# Evaluar ambas interpolaciones en los puntos comunes de A
T_values_left_interp_1_I = interp_left(A_common)

# Saving data to .npy files
np.save("A_common.npy", A_common)
np.save("T_values_left_interp_1_I.npy", T_values_left_interp_1_I)


# In[88]:


plt.figure(figsize=(10, 6))
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)
plt.plot(T_values_right_1[0:num_iterations_up_r], A_values_right_1[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right_1[num_iterations_up_r:], A_values_right_1[num_iterations_up_r:],color='blue')
plt.plot(T_values_right_1_p[0:num_iterations_up_r], A_values_right_1_p[0:num_iterations_up_r], color='blue')
plt.plot(T_values_right_1_p[num_iterations_up_r:], A_values_right_1_p[num_iterations_up_r:],color='blue')

plt.fill_betweenx(A_common, T_values_left_interp_1_I, T_values_right_interp_1_I, color='lightblue', alpha=0.5)

plt.xlabel(r'$T/T^*$')
plt.ylabel('A')
plt.title('Arnold tongues')
plt.grid(False)
plt.show()


# In[44]:


#FINALS ARNOLD TONGUES: k=2, k=infinity
import numpy as np

# Loading data back
A_common = np.load("A_common.npy")
T_values_left_interp_I = np.load("T_values_left_interp_I.npy")
T_values_right_interp_I = np.load("T_values_right_interp_I.npy")
T_values_left_interp_1_I = np.load("T_values_left_interp_1_I.npy")
T_values_right_interp_1_I = np.load("T_values_right_interp_1_I.npy")
T_values_left_interp_2_I = np.load("T_values_left_interp_2_I.npy")
T_values_right_interp_2_I = np.load("T_values_right_interp_2_I.npy")
T_left = np.load("T_values_left_interp_eii_1_2_inf.npy")
T_right = np.load("T_values_right_interp_eii_1_2_inf.npy")




time_points = np.linspace(0, Period, 1000)  # Sample points across one period
interp_values = interp_orbit_iPRC(time_points)  # Evaluate interp_orbit_iPRC over these points

# Extract components 1 and 5, and calculate their sum
Z_values = interp_values[1] + interp_values[5]  # Sum of component 1 and component 5

# Calculate Zmax and Zmin
Zmax = np.max(Z_values)
Zmin = np.min(Z_values)
print(Zmax)
print(Zmin)

# Define the range for A and calculate corresponding T/T* values
A_values = np.linspace(0, 0.2, 500)
T_over_Tstar_max = 1 + A_values * Zmax
T_over_Tstar_min = 1 + A_values * Zmin

T_over_Tstar_max_2 = (1/2)*(1 + A_values * Zmax)
T_over_Tstar_min_2 = (1/2)*(1 + A_values * Zmin)


T_over_Tstar_max = T_over_Tstar_max[T_over_Tstar_max > 0]
T_over_Tstar_min = T_over_Tstar_min[T_over_Tstar_min > 0]

T_over_Tstar_max_2 = T_over_Tstar_max_2[T_over_Tstar_max_2 > 0]
T_over_Tstar_min_2 = T_over_Tstar_min_2[T_over_Tstar_min_2 > 0]


# Plotting
plt.figure(figsize=(10, 6))
fill = np.full(len(A_values), 10)




plt.fill_betweenx(A_values[:len(T_over_Tstar_min)], 1 / T_over_Tstar_min[:len(T_over_Tstar_min)], 1 / T_over_Tstar_max[:len(T_over_Tstar_min)], color='lightblue', alpha=0.5,label=r'$1:1, \kappa \rightarrow \infty$')
plt.fill_betweenx(A_values[len(T_over_Tstar_min)+1:], 1 / T_over_Tstar_max[len(T_over_Tstar_min)+1:],fill[len(T_over_Tstar_min)+1:], color='lightblue', alpha=0.5)
# Plot the boundary lines for the filled region
plt.plot(1/T_over_Tstar_max, A_values[:len(T_over_Tstar_max)], color='lightblue')
plt.plot(1/T_over_Tstar_min, A_values[:len(T_over_Tstar_min)], color='lightblue')


plt.fill_betweenx(A_values[:len(T_over_Tstar_min_2)], 1 / T_over_Tstar_min_2[:len(T_over_Tstar_min_2)], 1 / T_over_Tstar_max_2[:len(T_over_Tstar_min_2)], color='pink', alpha=0.5,label=r'$2:1, \kappa \rightarrow \infty$')
plt.fill_betweenx(A_values[len(T_over_Tstar_min_2)+1:], 1 / T_over_Tstar_max_2[len(T_over_Tstar_min_2)+1:],fill[len(T_over_Tstar_min_2)+1:], color='pink', alpha=0.5)


# Plot the boundary lines for the filled region
plt.plot(1/T_over_Tstar_max_2, A_values[:len(T_over_Tstar_max_2)], color='pink')
plt.plot(1/T_over_Tstar_min_2, A_values[:len(T_over_Tstar_min_2)], color='pink')

plt.fill_betweenx(A_common, T_left, T_right, color='lightgreen', alpha=0.5,label=r'$1:2, \kappa \rightarrow \infty$')
plt.plot(T_left, A_common, color='lightgreen')
plt.plot(T_right, A_common, color='lightgreen')
# Gráfico de Arnold tongues y relleno entre curvas interpoladas 1:1
plt.plot(T_values_left_interp_I, A_common, color='blue')
plt.plot(T_values_right_interp_I, A_common, color='blue')
plt.fill_betweenx(A_common, T_values_left_interp_I, T_values_right_interp_I, color='blue',alpha=0.5,label=r'$1:1, \kappa=2$')



#Plot 1:2

plt.plot(T_values_left_interp_1_I, A_common, color='green')
plt.plot(T_values_right_interp_1_I, A_common, color='green')
plt.fill_betweenx(A_common, T_values_left_interp_1_I, T_values_right_interp_1_I, color='green',alpha=0.5,label=r'$1:2, \kappa=2$')


#Plot 2:1

plt.plot(T_values_left_interp_2_I, A_common, color='purple')
plt.plot(T_values_right_interp_2_I, A_common, color='purple')
plt.fill_betweenx(A_common, T_values_left_interp_2_I, T_values_right_interp_2_I, color='purple',alpha=0.5,label=r'$2:1, \kappa=2$')



# Setting limits for the axes
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)

# Labels and title
plt.xlabel(r'$\frac{T}{T^*}$')
plt.ylabel("A")

plt.grid(True)
plt.legend(loc='best')
plt.show()



# In[39]:


#Comparison between the obtained Arnold tongues and the original ones for k=2

import numpy as np

# Loading data back
A_common = np.load("A_common.npy")
T_values_left_interp_I = np.load("T_values_left_interp_I.npy")
T_values_right_interp_I = np.load("T_values_right_interp_I.npy")
T_values_left_interp_1_I = np.load("T_values_left_interp_1_I.npy")
T_values_right_interp_1_I = np.load("T_values_right_interp_1_I.npy")
T_values_left_interp_2_I = np.load("T_values_left_interp_2_I.npy")
T_values_right_interp_2_I = np.load("T_values_right_interp_2_I.npy")

T_values_left_interp = np.load("T_values_left_interp.npy")
T_values_right_interp = np.load("T_values_right_interp.npy")
T_values_left_interp_1 = np.load("T_values_left_interp_1.npy")
T_values_right_interp_1 = np.load("T_values_right_interp_1.npy")
T_values_left_interp_2 = np.load("T_values_left_interp_2.npy")
T_values_right_interp_2 = np.load("T_values_right_interp_2.npy")




plt.figure(figsize=(10, 6))


plt.plot(T_values_left_interp_I, A_common, color='blue')
plt.plot(T_values_right_interp_I, A_common, color='blue')
plt.fill_betweenx(A_common, T_values_left_interp_I, T_values_right_interp_I, color='blue',alpha=0.5,label=r'$1:1, \kappa=2$')



#Plot 1:2

plt.plot(T_values_left_interp_1_I, A_common, color='green')
plt.plot(T_values_right_interp_1_I, A_common, color='green')
plt.fill_betweenx(A_common, T_values_left_interp_1_I, T_values_right_interp_1_I, color='green',alpha=0.5,label=r'$1:2, \kappa=2$')


#Plot 2:1

plt.plot(T_values_left_interp_2_I, A_common, color='purple')
plt.plot(T_values_right_interp_2_I, A_common, color='purple')
plt.fill_betweenx(A_common, T_values_left_interp_2_I, T_values_right_interp_2_I, color='purple',alpha=0.5,label=r'$2:1, \kappa=2$')


plt.plot(T_values_left_interp, A_common, color='lightblue')
plt.plot(T_values_right_interp[1:], A_common[1:], color='lightblue')
plt.fill_betweenx(A_common[1:], T_values_left_interp[1:], T_values_right_interp[1:], color='lightblue',alpha=0.5,label=r'$1:1, \kappa=2$ Original')



#Plot 1:2

plt.plot(T_values_left_interp_1, A_common, color='lightgreen')
plt.plot(T_values_right_interp_1, A_common, color='lightgreen')
plt.fill_betweenx(A_common, T_values_left_interp_1, T_values_right_interp_1, color='lightgreen',alpha=0.5,label=r'$1:2, \kappa=2$ Original')


#Plot 2:1

plt.plot(T_values_left_interp_2, A_common, color='pink')
plt.plot(T_values_right_interp_2, A_common, color='pink')
plt.fill_betweenx(A_common, T_values_left_interp_2, T_values_right_interp_2, color='pink',alpha=0.5,label=r'$2:1, \kappa=2$ Original')



# Setting limits for the axes
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)

# Labels and title
plt.xlabel(r'$\frac{T}{T^*}$')
plt.ylabel("A")

plt.grid(True)
plt.legend(loc='best')
plt.show()


# ### CTC measures

# In[ ]:


#Pre computations of Delta tau
T=1.75*Period
A=0.07
t= np.linspace(0, 80 * T, 1000000) 
sol = odeint(neural_network_I, P_1, t, args =(A,T,), atol=atol, rtol=rtol)
P_2=sol[-1]
t= np.linspace(0, 2*T, 1000000)
sol=odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        
# Extract r_i(t) = x[5] and p(t)
r_i = sol[:, 4]
p_t = p(t, T)
        
# Ignore the initial transient by selecting only the times after transient_time
t_after_transient = t[(t <= T)]
#r_i_after_transient = r_i[t >= transient_time]
p_after_transient = p_t[(t <= T)]
        
# Find the time t_p of the maximum of p after the transient
t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
# Limit search for r_i to one period T after t_p
t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
# Find the maximum of r_i within this period
t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
t_inh = t_within_one_period[t_inh_index] 

print((t_inh - t_p) / T)
indices_after_4T = ((t >= t_p-7050)&(t<=t_p+7050))
# Optional: Plotting r_i and p to visualize
plt.figure(figsize=(12, 6))
plt.plot(t[indices_after_4T], r_i[indices_after_4T], label=r'$r_i(t)$ (inhibitory rate)')
plt.plot(t[indices_after_4T], A*p_t[indices_after_4T], label=r'$p(t)$ (external input)', linestyle='--')
plt.axvline(t_inh, color='red', linestyle=':', label=r'$t_{\text{inh}}$ (max $r_i$ after transient)')
plt.axvline(t_p, color='blue', linestyle=':', label=r'$t_p$ (max $p$ after transient)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Inhibitory Neuron Activity $r_i(t)$ and External Input $p(t)$ after Transient')
plt.legend()
plt.show()


# In[ ]:


#Computation of Delta tau
#We just save and plot the values of A and T, such that Delta tau is consistent through 5 consecutive periods

from scipy.interpolate import interp1d
t = np.linspace(0, 30 * Period, 100000)  # Adjust time range and resolution as needed
A_values= np.linspace(0.01,0.2,20)
# Prepare plot
# Initialize a list to store T_ratio_values for each A
T_ratio_matrix = []
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1,len(A_values)))

# Loop over each A
for i, A in enumerate(A_values):
    print(A)
    # Interpolate bounds for T based on A
    if A<0.075:
        T_left = interp_right(A)*Period
        T_right = interp_left(A)*Period
    else:
        T_left = interp_right(A)*Period
        T_right = 2.5*Period
    # Generate equidistant values of T within bounds
    T_values = np.linspace(T_left+0.003*Period, T_right-0.003*Period, 120)
    
    
    # Store Delta_tau values for current A
    delta_tau_values = []
    T_ratio_values = []
    # Listas para almacenar los grupos separados
    T_ratio_plus = []     # Para T_ratio con delta_tau > 0.6
    Delta_tau_plus = []   # Para delta_tau > 0.6
    T_ratio_minus = []    # Para T_ratio con delta_tau <= 0.6
    Delta_tau_minus = []  # Para delta_tau <= 0.6

    for T in T_values:
        # Simulate the system with current A and T
        t= np.linspace(0, 80 * T, 1000000) 
        sol = odeint(neural_network_I, P_1, t, args =(A,T,), atol=atol, rtol=rtol)
        P_2=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_1 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_3=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_3, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_2 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 2*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_4=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_4, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_3 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 3*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_4=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_4, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_3 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 3*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_5=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_5, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_4 = (t_inh - t_p) / T
        
        
        t_val= np.linspace(0, 4*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_6=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_6, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_5 = (t_inh - t_p) / T
        
        if ((abs(Delta_tau_1-Delta_tau_2)<0.01)&(abs(Delta_tau_1-Delta_tau_3)<0.01)&(abs(Delta_tau_1-Delta_tau_4)<0.01)&(abs(Delta_tau_1-Delta_tau_5)<0.01)):
            Delta_tau = (Delta_tau_1+Delta_tau_2+Delta_tau_3+Delta_tau_4+Delta_tau_5)/5
            T_ratio = T / Period
        
            # Store values for plotting
            delta_tau_values.append(Delta_tau)
            T_ratio_values.append(T_ratio)
        
        else: print(f'Not satisfied for T={T/Period}')
   
    for delta_tau, T_ratio in zip(delta_tau_values, T_ratio_values):
        if delta_tau > 0.6:
            Delta_tau_plus.append(delta_tau)
            T_ratio_plus.append(T_ratio)
        else:
            Delta_tau_minus.append(delta_tau)
            T_ratio_minus.append(T_ratio)

    T_ratio_matrix.append(T_ratio_values)
    # Plot Delta_tau vs T / Period for the current A
    plt.scatter(T_ratio_plus, Delta_tau_plus, label=f'A={A:.2f}', color=colors[i],s=1)
    plt.scatter(T_ratio_minus, Delta_tau_minus, color=colors[i],s=1)
    
# Label and show plot
plt.xlabel(r'$T / T^*$')
plt.ylabel(r'$\Delta_{\tau}$')
plt.legend(loc='upper left')
plt.title(r'$\Delta_{\tau}$ vs $T / T^*$ for Different Values of A')
plt.grid(True)
plt.show()


# In[ ]:


from scipy.interpolate import interp1d
t = np.linspace(0, 30 * Period, 100000)  # Adjust time range and resolution as needed
A_values= np.linspace(0.01,0.1,10)
# Prepare plot
# Initialize a list to store T_ratio_values for each A
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1,len(A_values)))

# Loop over each A
for i, A in enumerate(A_values):
    print(A)
    T_values = filtered_T_ratio_matrix[i]
    
    
    # Store Delta_tau values for current A
    delta_tau_values = []
    T_ratio_values = []
    # Listas para almacenar los grupos separados
    T_ratio_plus = []     # Para T_ratio con delta_tau > 0.6
    Delta_tau_plus = []   # Para delta_tau > 0.6
    T_ratio_minus = []    # Para T_ratio con delta_tau <= 0.6
    Delta_tau_minus = []  # Para delta_tau <= 0.6

    for T in T_values:
        T=Period*T
        print(T)
        # Simulate the system with current A and T
        t= np.linspace(0, 80 * T, 1000000) 
        sol = odeint(neural_network_I, P_1, t, args =(A,T,), atol=atol, rtol=rtol)
        P_2=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_1 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_3=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_3, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_2 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 2*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_4=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_4, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_3 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 3*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_4=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_4, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_3 = (t_inh - t_p) / T
        
        t_val= np.linspace(0, 3*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_5=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_5, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_4 = (t_inh - t_p) / T
        
        
        t_val= np.linspace(0, 4*T, 1000000) 
        sol = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        P_6=sol[-1]
        t= np.linspace(0, 2*T, 1000000)
        sol=odeint(neural_network_I, P_6, t, args =(A,T,), atol=atol, rtol=rtol)
        
        # Extract r_i(t) = x[5] and p(t)
        r_i = sol[:, 4]
        p_t = p(t, T)
        
        # Ignore the initial transient by selecting only the times after transient_time
        t_after_transient = t[(t <= T)]
        #r_i_after_transient = r_i[t >= transient_time]
        p_after_transient = p_t[(t <= T)]
        
        # Find the time t_p of the maximum of p after the transient
        t_p_index = np.argmax(p_after_transient)  # Index of max p after transient
        t_p = t_after_transient[t_p_index]        # Time at which p reaches maximum
        
        
        # Limit search for r_i to one period T after t_p
        t_within_one_period = t[(t>=t_p)&(t<t_p+T)]
        r_i_within_one_period = r_i[(t>=t_p)&(t<t_p+T)]
        
        # Find the maximum of r_i within this period
        t_inh_index = np.argmax(r_i_within_one_period)  # Index of max r_i within one period after t_p
        t_inh = t_within_one_period[t_inh_index]        # Time at which r_i reaches max within one period after t_p
        
        # Calculate Delta_tau
        Delta_tau_5 = (t_inh - t_p) / T
        
        Delta_tau = (Delta_tau_1+Delta_tau_2+Delta_tau_3+Delta_tau_4+Delta_tau_5)/5
        T_ratio = T / Period
        
        # Store values for plotting
        delta_tau_values.append(Delta_tau)
        T_ratio_values.append(T_ratio)
        

    for delta_tau, T_ratio in zip(delta_tau_values, T_ratio_values):
        if delta_tau > 0.6:
            Delta_tau_plus.append(delta_tau)
            T_ratio_plus.append(T_ratio)
        else:
            Delta_tau_minus.append(delta_tau)
            T_ratio_minus.append(T_ratio)

    # Plot Delta_tau vs T / Period for the current A
    plt.plot(T_ratio_plus, Delta_tau_plus, label=f'A={A:.2f}', color=colors[i])
    plt.plot(T_ratio_minus, Delta_tau_minus, color=colors[i])
    
# Label and show plot
plt.xlabel(r'$T / T^*$')
plt.ylabel(r'$\Delta_{\tau}$')
plt.legend(loc='upper left')
plt.title(r'$\Delta_{\tau}$ vs $T / T^*$ for Different Values of A')
plt.grid(True)
plt.show()


# In[ ]:


#Save the matrix containing the A and T/T^* values that we consider as "the good Arnold tongue"
import pickle

# Save inhomogeneous data
with open("T_ratio_matrix.pkl", "wb") as f:
    pickle.dump(T_ratio_matrix, f)

# Load it back
with open("T_ratio_matrix.pkl", "rb") as f:
    loaded_data = pickle.load(f)


# In[ ]:


import pickle
# Load it back
with open("T_ratio_matrix.pkl", "rb") as f:
    loaded_data = pickle.load(f)


# In[ ]:


#Plot the values where the phase reduction aligns with the global dynamics
import numpy as np
A_values = np.linspace(0.01, 0.2, 20) 
# Loading data back
A_common = np.load("A_common.npy")
T_values_left_interp_I = np.load("T_values_left_interp_I.npy")
T_values_right_interp_I = np.load("T_values_right_interp_I.npy")
T_values_left_interp_1_I = np.load("T_values_left_interp_1_I.npy")
T_values_right_interp_1_I = np.load("T_values_right_interp_1_I.npy")
T_values_left_interp_2_I = np.load("T_values_left_interp_2_I.npy")
T_values_right_interp_2_I = np.load("T_values_right_interp_2_I.npy")

# Plotting
plt.figure(figsize=(10, 6))
fill = np.full(len(A_values), 10)

plt.plot(T_values_left_interp_I, A_common, color='blue')
plt.plot(T_values_right_interp_I, A_common, color='blue')
plt.fill_betweenx(A_common, T_values_left_interp_I, T_values_right_interp_I, color='blue',alpha=0.5,label=r'$1:1, \kappa=2$')

A_values = np.linspace(0.01, 0.2, 20) 

for i in range(len(T_ratio_matrix)):
    length = len(T_ratio_matrix[i])
    ones = (i*0.01+0.01) * np.ones(length)
    plt.scatter(T_ratio_matrix[i], ones, color='red', s=1)  # Desplazar ligeramente las y para cada i




# Setting limits for the axes
plt.xlim(0, 2.5)
plt.ylim(0, 0.2)

# Labels and title
plt.xlabel(r'$\frac{T}{T^*}$')
plt.ylabel("A")

plt.grid(True)
plt.legend(loc='best')
plt.show()


# In[ ]:


#Define the new Arnold Tongue
threshold = 0.015

# Función para filtrar puntos antes de un salto
def filter_until_jump(series, threshold):
    for i in range(len(series) - 1):
        if abs(series[i + 1] - series[i]) > threshold:
            return series[:i + 1]  # Devuelve hasta el último punto antes del salto
    return series  # Si no hay salto, devuelve toda la serie

# Aplicar la función a cada conjunto de puntos en T_ratio_matrix
filtered_T_ratio_matrix = [filter_until_jump(series, threshold) for series in T_ratio_matrix]

# Mostrar resultados
for i, series in enumerate(filtered_T_ratio_matrix):
    print(f"Filtered series {i}: {series}")


# In[ ]:


filtered_series_01 = [
    [np.float64(0.9521116356499515), np.float64(0.9535263488150016), np.float64(0.9549410619800519), np.float64(0.956355775145102), np.float64(0.9577704883101522), np.float64(0.9591852014752024), np.float64(0.9605999146402526), np.float64(0.9620146278053028), np.float64(0.9634293409703529), np.float64(0.9648440541354031), np.float64(0.9662587673004532), np.float64(0.9676734804655035), np.float64(0.9690881936305537), np.float64(0.9705029067956038), np.float64(0.971917619960654), np.float64(0.9733323331257042), np.float64(0.9747470462907544), np.float64(0.9761617594558046), np.float64(0.9775764726208548), np.float64(0.9789911857859048), np.float64(0.9804058989509551), np.float64(0.9818206121160052), np.float64(0.9832353252810555), np.float64(0.9846500384461058), np.float64(0.9860647516111557), np.float64(0.987479464776206), np.float64(0.9888941779412562), np.float64(0.9903088911063063), np.float64(0.9917236042713565), np.float64(0.9931383174364067), np.float64(0.9945530306014569), np.float64(0.995967743766507), np.float64(0.9973824569315572), np.float64(0.9987971700966075), np.float64(1.0002118832616578), np.float64(1.0016265964267077), np.float64(1.003041309591758), np.float64(1.0044560227568082), np.float64(1.0058707359218582), np.float64(1.0072854490869085), np.float64(1.0087001622519587), np.float64(1.010114875417009), np.float64(1.011529588582059), np.float64(1.0129443017471091), np.float64(1.0143590149121595), np.float64(1.0157737280772097), np.float64(1.0171884412422598), np.float64(1.01860315440731), np.float64(1.0200178675723601), np.float64(1.0214325807374103), np.float64(1.0228472939024604), np.float64(1.0242620070675108), np.float64(1.025676720232561), np.float64(1.0270914333976109), np.float64(1.0285061465626613), np.float64(1.0299208597277114), np.float64(1.0313355728927616), np.float64(1.0327502860578117), np.float64(1.0341649992228619), np.float64(1.0355797123879122), np.float64(1.0369944255529622), np.float64(1.0384091387180125), np.float64(1.0398238518830627), np.float64(1.0412385650481126), np.float64(1.042653278213163), np.float64(1.0440679913782132), np.float64(1.0454827045432635), np.float64(1.0468974177083135), np.float64(1.0483121308733636), np.float64(1.049726844038414), np.float64(1.0511415572034641), np.float64(1.0525562703685143), np.float64(1.0539709835335644), np.float64(1.0553856966986146), np.float64(1.056800409863665), np.float64(1.058215123028715), np.float64(1.0596298361937653)
    ],
    [
       np.float64(0.9220767118016707), np.float64(0.9254844531837897), np.float64(0.9288921945659085), np.float64(0.9322999359480275), np.float64(0.9357076773301463), np.float64(0.9391154187122652), np.float64(0.9425231600943842), np.float64(0.945930901476503), np.float64(0.9493386428586219), np.float64(0.9527463842407408), np.float64(0.9561541256228597), np.float64(0.9595618670049786), np.float64(0.9629696083870974), np.float64(0.9663773497692164), np.float64(0.9697850911513354), np.float64(0.9731928325334541), np.float64(0.9766005739155731), np.float64(0.980008315297692), np.float64(0.9834160566798108), np.float64(0.9868237980619298), np.float64(0.9902315394440486), np.float64(0.9936392808261676), np.float64(0.9970470222082864), np.float64(1.0004547635904055), np.float64(1.0038625049725243), np.float64(1.007270246354643), np.float64(1.010677987736762), np.float64(1.014085729118881), np.float64(1.017493470501), np.float64(1.0209012118831187), np.float64(1.0243089532652376), np.float64(1.0277166946473566), np.float64(1.0311244360294753), np.float64(1.0345321774115943), np.float64(1.037939918793713), np.float64(1.041347660175832), np.float64(1.044755401557951), np.float64(1.04816314294007), np.float64(1.051570884322189), np.float64(1.0549786257043077), np.float64(1.0583863670864264), np.float64(1.0617941084685454), np.float64(1.0652018498506644), np.float64(1.0686095912327833), np.float64(1.072017332614902), np.float64(1.075425073997021), np.float64(1.0788328153791398), np.float64(1.0822405567612587), np.float64(1.0856482981433777)
    ],
    [
     np.float64(0.879073215082744), np.float64(0.8881039987314341), np.float64(0.8971347823801242), np.float64(0.9061655660288144), np.float64(0.9151963496775044), np.float64(0.9242271333261945), np.float64(0.9332579169748847), np.float64(0.9422887006235747), np.float64(0.9513194842722649), np.float64(0.960350267920955), np.float64(0.9693810515696452), np.float64(0.9784118352183352), np.float64(0.9874426188670254), np.float64(0.9964734025157154), np.float64(1.0055041861644054), np.float64(1.0145349698130957), np.float64(1.0235657534617857), np.float64(1.032596537110476), np.float64(1.0416273207591658), np.float64(1.050658104407856), np.float64(1.059688888056546), np.float64(1.0687196717052363), np.float64(1.0777504553539263), np.float64(1.0867812390026166), np.float64(1.0958120226513066), np.float64(1.1048428062999966), np.float64(1.1138735899486867), np.float64(1.1229043735973767)
    ],
    [
    np.float64(0.8686358613453208), np.float64(0.8795859696259333), np.float64(0.8905360779065458), np.float64(0.9014861861871585), np.float64(0.912436294467771), np.float64(0.9233864027483835), np.float64(0.934336511028996), np.float64(0.9452866193096086), np.float64(0.956236727590221), np.float64(0.9671868358708335), np.float64(0.9781369441514461), np.float64(0.9890870524320587), np.float64(1.0000371607126712), np.float64(1.0109872689932837), np.float64(1.0219373772738962), np.float64(1.0328874855545087), np.float64(1.0438375938351214), np.float64(1.054787702115734), np.float64(1.0657378103963464), np.float64(1.076687918676959), np.float64(1.0876380269575716), np.float64(1.0985881352381839), np.float64(1.1095382435187966), np.float64(1.1204883517994093), np.float64(1.1314384600800216), np.float64(1.1423885683606343)
    ],
    [np.float64(0.8628820837690917), np.float64(0.8752017106465797), np.float64(0.8875213375240676), np.float64(0.8998409644015555), np.float64(0.9121605912790435), np.float64(0.9244802181565314), np.float64(0.9367998450340193), np.float64(0.9491194719115073), np.float64(0.9614390987889951), np.float64(0.9737587256664831), np.float64(0.986078352543971), np.float64(0.9983979794214589), np.float64(1.010717606298947), np.float64(1.023037233176435), np.float64(1.035356860053923), np.float64(1.0476764869314108), np.float64(1.0599961138088987), np.float64(1.0723157406863866), np.float64(1.0846353675638745), np.float64(1.0969549944413624), np.float64(1.1092746213188505), np.float64(1.1215942481963384), np.float64(1.1339138750738262), np.float64(1.1462335019513141), np.float64(1.1585531288288022), np.float64(1.1708727557062901)],
    [np.float64(0.8479207550026641), np.float64(0.8613248823813), np.float64(0.8747290097599361), np.float64(0.888133137138572), np.float64(0.9015372645172078), np.float64(0.9149413918958438), np.float64(0.9283455192744798), np.float64(0.9417496466531158), np.float64(0.9551537740317517), np.float64(0.9685579014103878), np.float64(0.9819620287890237), np.float64(0.9953661561676597), np.float64(1.0087702835462957), np.float64(1.0221744109249316), np.float64(1.0355785383035676), np.float64(1.0489826656822037), np.float64(1.0623867930608393), np.float64(1.0757909204394753), np.float64(1.0891950478181114), np.float64(1.1025991751967472), np.float64(1.1160033025753833), np.float64(1.1294074299540193), np.float64(1.1428115573326552), np.float64(1.1562156847112912), np.float64(1.1696198120899273), np.float64(1.183023939468563)],
    [np.float64(0.8492127095333892), np.float64(0.8630596615541171), np.float64(0.8769066135748449), np.float64(0.8907535655955726), np.float64(0.9046005176163004), np.float64(0.9184474696370283), np.float64(0.9322944216577561), np.float64(0.9461413736784839), np.float64(0.9599883256992118), np.float64(0.9738352777199395), np.float64(0.9876822297406672), np.float64(1.0015291817613954), np.float64(1.015376133782123), np.float64(1.0292230858028508), np.float64(1.0430700378235787), np.float64(1.0569169898443065), np.float64(1.0707639418650343), np.float64(1.0846108938857622), np.float64(1.09845784590649), np.float64(1.1123047979272178), np.float64(1.1261517499479456), np.float64(1.1399987019686733), np.float64(1.1538456539894013), np.float64(1.167692606010129), np.float64(1.1815395580308568), np.float64(1.1953865100515846), np.float64(1.2092334620723124)],
     [np.float64(0.854015928082426), np.float64(0.8678225169220695), np.float64(0.8816291057617129), np.float64(0.8954356946013564), np.float64(0.909242283441), np.float64(0.9230488722806434), np.float64(0.936855461120287), np.float64(0.9506620499599303), np.float64(0.9644686387995738), np.float64(0.9782752276392174), np.float64(0.9920818164788608), np.float64(1.0058884053185042), np.float64(1.0196949941581477), np.float64(1.0335015829977912), np.float64(1.0473081718374346), np.float64(1.0611147606770783), np.float64(1.0749213495167218), np.float64(1.088727938356365), np.float64(1.1025345271960085), np.float64(1.116341116035652), np.float64(1.1301477048752957), np.float64(1.1439542937149392), np.float64(1.1577608825545824), np.float64(1.171567471394226), np.float64(1.1853740602338694), np.float64(1.199180649073513), np.float64(1.2129872379131565), np.float64(1.2267938267527998)],
    [np.float64(0.861979293824537), np.float64(0.8757189636243309), np.float64(0.8894586334241246), np.float64(0.9031983032239185), np.float64(0.9169379730237123), np.float64(0.930677642823506), np.float64(0.9444173126232999), np.float64(0.9581569824230937), np.float64(0.9718966522228876), np.float64(0.9856363220226814), np.float64(0.999375991822475), np.float64(1.0131156616222687), np.float64(1.0268553314220628), np.float64(1.0405950012218566), np.float64(1.0543346710216501), np.float64(1.0680743408214441), np.float64(1.081814010621238), np.float64(1.0955536804210317), np.float64(1.1092933502208255), np.float64(1.1230330200206193), np.float64(1.1367726898204131), np.float64(1.150512359620207), np.float64(1.1642520294200007), np.float64(1.1779916992197947), np.float64(1.1917313690195883), np.float64(1.2054710388193821), np.float64(1.2192107086191761), np.float64(1.2329503784189697), np.float64(1.2466900482187635)],
    [np.float64(0.8727848976859471), np.float64(0.8864337640919475), np.float64(0.900082630497948), np.float64(0.9137314969039484), np.float64(0.9273803633099489), np.float64(0.9410292297159493), np.float64(0.9546780961219498), np.float64(0.9683269625279503), np.float64(0.9819758289339507), np.float64(0.995624695339951), np.float64(1.0092735617459516), np.float64(1.022922428151952), np.float64(1.0365712945579524), np.float64(1.050220160963953), np.float64(1.0638690273699534), np.float64(1.0775178937759538), np.float64(1.0911667601819541), np.float64(1.1048156265879545), np.float64(1.1184644929939551), np.float64(1.1321133593999555), np.float64(1.145762225805956), np.float64(1.1594110922119565), np.float64(1.1730599586179569), np.float64(1.1867088250239572), np.float64(1.2003576914299576), np.float64(1.2140065578359582), np.float64(1.2276554242419586), np.float64(1.241304290647959), np.float64(1.2549531570539596)]
]


# In[ ]:


#Plot the "new" Arnold tongue
# Plotting
plt.figure(figsize=(10, 6))
A_values = np.linspace(0, 0.1, 11) 
fill = np.full(len(A_values), 10)
last_points = [row[-1] if len(row) > 0 else np.nan for row in filtered_series_01]
# Añadir el valor 0 como primer elemento de la lista
last_points = [1] + last_points
# Crear un conjunto de puntos comunes para el eje y (A), por ejemplo, de 0 a 0.2
A_common = np.linspace(0, 0.2, 200)

# Interpolar ambas curvas para estos puntos comunes en función de A
interp_left = interp1d(A_values, last_points, bounds_error=False, fill_value="extrapolate")
Interp_left= interp_left(A_common)
# Evaluar ambas interpolaciones en los puntos comunes de A
T_values_left_interp = interp_left(A_common)
T_values_right_interp = interp_right(A_common)


# Gráfico de Arnold tongues y relleno entre curvas interpoladas 1:1
plt.plot(T_values_left_interp_I, A_common, color='blue')
plt.plot(T_values_right_interp_I, A_common, color='blue')
plt.fill_betweenx(A_common, T_values_left_interp_I, T_values_right_interp_I, color='blue',alpha=0.5,label=r'$1:1, \kappa=2$')
plt.fill_betweenx(A_common, T_values_right_interp_I, Interp_left, color='red', alpha=0.5)


for i in range(len(filtered_series_01)):
    length = len(filtered_series_01[i])
    ones = (i*0.01+0.01) * np.ones(length)
    plt.scatter(filtered_series_01[i], ones, color='red', s=1)  # Desplazar ligeramente las y para cada i




# Setting limits for the axes
plt.xlim(0, 2.5)
plt.ylim(0, 0.1)

# Labels and title
plt.xlabel(r'$\frac{T}{T^*}$')
plt.ylabel("A")

plt.grid(True)
plt.legend(loc='best')
plt.show()


# In[ ]:


#Computation of Delta Bar Alpha (considering just the "new" Arnold tongue)
import numpy as np
from scipy.integrate import odeint, trapezoid
import matplotlib.pyplot as plt

t = np.linspace(0, 40 * Period,1000000)
sol_0 = odeint(neural_network_I, P_1, t, args =(0,1,), atol=atol, rtol=rtol)
P_2=sol_0[-1]
t = np.linspace(0,  Period,10000)
sol_0 = odeint(neural_network_I, P_2, t, args =(0,1,), atol=atol, rtol=rtol)
r_e_0 = sol_0[:,0]  # Assuming r_e is the first component of x; update as necessary
# Compute R_bar_0 (mean over one period T_star)
R_bar_0 = trapezoid(r_e_0, dx=Period/len(t)) / Period
# Setup parameters for plotting
A_values = np.linspace(0.01, 0.1, 10)  # Range of perturbation values
colors = plt.cm.viridis(np.linspace(0, 1, len(A_values)))

plt.figure(figsize=(12, 8))

for i, A in enumerate(A_values):
    print(A)
    
    T_values = filtered_T_ratio_matrix[i]
    delta_bar_alpha_values = []
    T_ratio_values = []

    for T in T_values:
        T=T*Period
        print(T)
        # Solve system with perturbation A
        t = np.linspace(0, 40 * T,1000000)
        sol_A = odeint(neural_network_I, P_1, t, args =(A,T,), atol=atol, rtol=rtol)
        P_2=sol_A[-1]
        t = np.linspace(0, T,1000000)
        sol_A = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        r_e_A=sol_A[:,0]
        R_bar_A = trapezoid(r_e_A, dx=T/len(t)) / T
        #print(R_bar_A)
        # Compute Delta_bar_alpha
        Delta_bar_alpha = R_bar_A / R_bar_0
        T_ratio = T / Period

        # Store values for plotting
        delta_bar_alpha_values.append(Delta_bar_alpha)
        T_ratio_values.append(T_ratio)

    plt.plot(T_ratio_values, delta_bar_alpha_values, label=f'A={A:.2f}', color=colors[i])

# Label and show plot
plt.xlabel(r'$T / T^*$')
plt.ylabel(r'$\Delta_{\bar{\alpha}}$')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[2]:


#Computation of Delta alpha (considering just the "new" Arnold tongue)
t = np.linspace(0, 80 * Period, 1000000)  # Adjust time range and resolution as needed

sol_0 = odeint(neural_network_I, P_1, t, args =(0,1,), atol=atol, rtol=rtol)
t = np.linspace(0, Period, 1000000)  # Adjust time range and resolution as needed
P_2=sol_0[-1]
sol_0 = odeint(neural_network_I, P_2, t, args =(0,1,), atol=atol, rtol=rtol)
r_e_0 = sol_0[:,0]  # Assuming r_e is the first component of x; update as necessary

# Compute R_bar_0 (mean over one period T_star)
R_0 = np.max(r_e_0)

A_values= np.linspace(0.01,0.1,10)
# Prepare plot
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(A_values)))

# Loop over each A
for i, A in enumerate(A_values):
    # Interpolate bounds for T based on A
     # Interpolate bounds for T based on A
    print(A)
    
    T_values = filtered_T_ratio_matrix[i]

    delta_alpha_values = []
    T_ratio_values = []

    for T in T_values:
        T=T*Period
        print(T)
        # Solve system with perturbation A
        t=np.linspace(0, 80 * T, 10000000)
        sol_A = odeint(neural_network_I, P_1, t, args =(A,T,), atol=atol, rtol=rtol)
        #plt.plot(t,sol_A[:,0],label=f'{T}')
        P_2=sol_A[-1]
        t=np.linspace(0, T, 1000000)
        sol_A = odeint(neural_network_I, P_2, t, args =(A,T,), atol=atol, rtol=rtol)
        r_e_A=sol_A[:,0]
        R_A = np.max(r_e_A)
        
        #print(R_bar_A)
        # Compute Delta_bar_alpha
        Delta_alpha = R_A / R_0
        T_ratio = T / Period

        # Store values for plotting
        delta_alpha_values.append(Delta_alpha)
        T_ratio_values.append(T_ratio)

    plt.plot(T_ratio_values, delta_alpha_values, label=f'A={A:.2f}', color=colors[i])
    
# Label and show plot
plt.xlabel(r'$T / T^*$')
plt.ylabel(r'$\Delta_{\alpha}$')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[3]:


#Computation of delta sigma (considering just the "new" Arnold tongue)
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define time array and solve for unperturbed solution
t = np.linspace(0,40 * Period, 100000)  # Adjust time range and resolution as needed
sol_0 = odeint(neural_network_I, P_1, t, args=(0, 1), atol=atol, rtol=rtol)
P_2=sol_0[-1]
t = np.linspace(0, 2*Period, 10000) 
sol_0 = odeint(neural_network_I, P_2, t, args=(0, 1), atol=atol, rtol=rtol)
r_e_0 = sol_0[:, 0]  # Assuming r_e is the first component of x

t_min=(t<=Period)
r_e_0_min=r_e_0[t_min]
index_min = np.argmin(r_e_0_min)
t_0 = t[index_min]


mask_range = (t >= t_0) & (t < t_0 + Period)
t_in_range = t[mask_range]
r_e_in_range = r_e_0[mask_range]



# Define a function to find t1 and t2 for half-width calculation
def find_half_width_times(r_e, t_interval):
    # Calculate the threshold as half of max + min of r_e in the interval
    r_e_max = np.max(r_e)
    r_e_min = np.min(r_e)
    threshold = 0.5 * (r_e_max + r_e_min)
    
    # Find times where r_e crosses the threshold
    indices_above = np.where(r_e >= threshold)[0]
    
    # Ensure at least two times are found
    if len(indices_above) >= 2:
        t1 = t_interval[indices_above[0]]
        t2 = t_interval[indices_above[-1]]
        return 0.5 * (t2 - t1)
    else:
        return None

HW_0=find_half_width_times(r_e_in_range,t_in_range)
A_values = np.linspace(0.01, 0.1, 10)  # Define your A values

# Prepare plot
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(A_values)))

# Loop over each A value to compute and plot Delta_omega
for i, A in enumerate(A_values):
    print(A)
    T_values = filtered_T_ratio_matrix[i]
    delta_omega_values = []
    T_ratio_values = []

    for T in T_values:
        T=T*Period
        # Solve system with perturbation A
        t = np.linspace(0, 80 * T, 500000)  # Adjust time range and resolution as needed
        sol_A = odeint(neural_network_I, P_1, t, args=(A, T), atol=atol, rtol=rtol)
        P_2=sol_A[-1]
        t = np.linspace(0, 2*Period, 1000000) 
        sol_A = odeint(neural_network_I, P_2, t, args=(A, T), atol=atol, rtol=rtol)
        r_e_A = sol_A[:, 0]  # Assuming r_e is the first component of x
        
        t_min=(t<=T)
        r_e_A_min=r_e_A[t_min]
        index_min = np.argmin(r_e_A_min)
        t_0 = t[index_min]

       
        mask_range = (t >= t_0) & (t < t_0 + T)
        t_in_range = t[mask_range]
        r_e_in_range = r_e_A[mask_range]
        
        HW_A=find_half_width_times(r_e_in_range,t_in_range)

        # Only proceed if HW_A was successfully computed
        if HW_A is not None and HW_0 is not None:
            Delta_omega = (HW_A / T) / (HW_0 / Period)
            T_ratio = T / Period

            delta_omega_values.append(Delta_omega)
            T_ratio_values.append(T_ratio)

    # Plot Delta_omega vs T / Period for the current A
    plt.plot(T_ratio_values, delta_omega_values, label=f'A={A:.2f}', color=colors[i])

# Label and show plot
plt.xlabel(r'$T / \text{T}^*$')
plt.ylabel(r'$\Delta_{\sigma}$')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


# In[ ]:




