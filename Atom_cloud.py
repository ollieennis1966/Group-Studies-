from Vector import Vector

import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, pi, exp, log10, sin, cos

class Cloud:
    def __init__(self, Temp_0 = 1e-5, N = int(1e4), diameter = 1e-3):
        """
        Initial conditions of the atom cloud: i.e. at time t = 0 when the MOT is turned off.
        Assumptions: no interatomic collisions, cloud is uniform.
        
        Initial radius, spatial and velocity distributions and temperature.
        """
        hbar = 1.055e-34 # Planck's reduced constant
        c = 3e8 # Speed of light in a vacuum

        k = 1.38e-23 # Boltzmann constant
        M = 1.443e-25 # Mass of Rb-87 atom
        mean_velocity = sqrt((8*k*Temp_0)/(pi*M)) # Mean atom speed

        radius = diameter/2 # Cloud radius
        
        # Speed, velocity and spatial distributions
        V, R = [], []
        for i in range(0, N):
            
            v = random.uniform(1e-5, 1) # Atom speed in MOT
            Prob = ((M/(2*pi*k*Temp_0))**(3/2))*(4*pi*v**2)*exp(-(M*v**2)/(2*k*Temp_0)) # Boltzmann distribution function of atom speed
            v *= Prob # Weighted atom speed
            
            theta, phi = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles
            
            vx, vy, vz = v*sin(theta)*cos(phi), v*sin(theta)*sin(phi), v*cos(theta) # Velocity components
            velocity = Vector(vx, vy, vz) # Velocity vector
            V.append(velocity)

            x, y, z = radius*random.uniform(-1, 1), radius*random.uniform(-1, 1), radius*random.uniform(-1, 1) # Position components
            position = Vector(x, y, z) # Position vector
            R.append(position)

        self.k, self.M, self.N, self.hbar, self.c = k, M, N, hbar, c
        self.velocity, self.position = V, R


    def fall_expansion(self, g, dv, dt, n):
        """
        Evolution of atom cloud spatial and velocity distributions after MOT is turned off,
        due to gravity and thermal motion.
        """
        velocity, position = self.velocity, self.position
        Mean_V, Max_R = [], []

        for j in range(0, n):
            for i in range(0, self.N):
                position[i] += velocity[i]*dt # Displacement of atom after time dt
                velocity[i] += dv*Vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) # Change in atom velocity caused by thermal motion
                velocity[i][2] += g*dt # Change in atom speed along z-axis due to gravity
                    
                Mean_V.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
                Max_R.append(sqrt(position[i][0]**2 + position[i][1]**2))
            
            mean_velocity = np.mean(Mean_V) # Mean atom speed after time dt
            Temp = ((pi*self.M)/(8*self.k))*mean_velocity**2 # Temperature of atom cloud after time dt
            radius = np.max(Max_R) # Radius of atom cloud in x-y plane after time dt
        
        return mean_velocity, radius, velocity, position


    def gaussian(self, I0, w0, l, z, r):
        """
        Alignment of Raman beams with local gravity
        Gaussian profile of Raman beams
        """
        w = sqrt(w0**2 + ((l*z)/(pi/w0)**2))
        I = I0*exp(-2*(r/w**2))
        return I


    def light_atom(self, i, theta1, theta2, omega, pulse):
        """
        Momentum recoil exerted on atom by two-photon transition
        """
        w1, w2 = 384.2e12 + 4.271e9, 384.2e12 - 2.563e9 # Frequencies of beam photons
        
        k1, k2 = 2*pi*(w1/self.c), 2*pi*(w2/self.c) # wavevectors of beam photons

        if i == -1: k_eff = k1*cos(theta1) - k2*cos(theta2) # Co-propagating beams
        elif i == 1: k_eff = k1 + k2 # Counter-propagating beams

        dp = self.hbar*k_eff # Momentum recoil of atom after two-photon transition
        dv = dp/self.M # Change in z-axis speed due to atom-photon interactions

        if pulse == 1: # 1st Raman pulse
            cycle = 1/4
            x = cloud_0
            velocity, position = x[2], x[3] # Atom velocities and positions before 1st pulse
        
        elif pulse == 2: # 2nd Raman pulse
           cycle = 1/2
           x = pulse1
           velocity, position = x[0], x[1] # Atom velocities and positions before 2nd pulse
        
        elif pulse == 3: # 3rd Raman pulse
            cycle = 1/4
            x = pulse2
            velocity, position = x[0], x[1] # Atom velocities and positions before 3rd pulse

        tau = ((2*pi)/omega)*cycle # Raman pulse length

        pulse_length = np.linspace(0, tau, 100)
        Prob_1 = np.sin(0.5*omega*pulse_length)**2 # Rabi oscillation
        Prob_final = np.sin(0.5*omega*tau)**2 # Proportion of excited atoms at end of Raman pulse

        plt.figure(figsize = (6, 4))
        plt.plot(pulse_length, Prob_1, label = "P(Excitation)")
        plt.xlabel("time")
        plt.ylabel("Probability(time)")
        plt.legend(loc = 'upper right')
        plt.show()

        for j in range(0, int(Prob_final*self.N)): # Fraction of atoms excited by Raman pulse
            velocity[j][2] -= dv
            position[j][2] -= dv*tau
        
        return velocity, position


if __name__ == "__main__":
    C = Cloud()

    cloud_0 = C.fall_expansion(9.81, 1e-2, 1e-3, 10)
    beam = C.gaussian(32, 2e-2, 780e-9, 1e-3, 1e-3)

    pulse1 = C.light_atom(1, 0, 0, 1e6, 1)
    pulse2 = C.light_atom(1, 0, 0, 1e6, 2)
    pulse3 = C.light_atom(1, 0, 0, 1e6, 3)
