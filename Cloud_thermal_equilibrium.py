from Vector import Vector

import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, pi, exp, log10, sin, cos

N = int(1e4) # Number of atoms in cloud
M = 1.443e-25 # Mass of Rb-87 atom
hbar = 1.055e-34 # Planck's reduced constant
c = 3e8 # Speed of light in a vacuum
k = 1.38e-23 # Boltzmann constant
epsilon = 8.85e-12 # Vacuum permittivity
Temp_0 = 1e-5


def initial_cloud(diameter = 1e-3):
    """
    Initial conditions of the atom cloud: i.e. at time t = 0 when the MOT is turned off.
    Assumptions: no interatomic collisions, uniform cloud.
        
    Initial radius, spatial and velocity distributions and temperature.
    """
    radius = diameter/2 # Cloud radius at time t = 0
        
    # Speed, velocity and spatial distributions
    Velocity, Position, Speed, Distance = [], [], [], []

    XY = []

    for i in range(0, N):
            
        v = random.uniform(0, 1)
        speed_distribution = ((M/(2*pi*k*Temp_0))**(3/2))*(4*pi*v**2)*exp(-(M*v**2)/(2*k*Temp_0)) # Boltzmann distribution of atom speed
        v *= speed_distribution # Weighted atom speed
        Speed.append(v)
            
        theta_v, phi_v = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom velocity vector
            
        vx, vy, vz = v*sin(theta_v)*cos(phi_v), v*sin(theta_v)*sin(phi_v), v*cos(theta_v) # Velocity components
        velocity = Vector(vx, vy, vz) # Velocity vector
        Velocity.append(velocity)

        r = radius*random.random()
        position_distribution = exp(-r**2) # Gaussian distribution of atom distance
        r*= position_distribution # Weighted atom distance from cloud center
        Distance.append(r)

        theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

        x, y, z = r*sin(theta_r)*cos(phi_r), r*sin(theta_r)*sin(phi_r), r*cos(theta_r) # Position components
        position = Vector(x, y, z) # Position vector
        Position.append(position)

        XY.append(sqrt(x**2 + y**2)) # Distance of atom from cenetr of cloud in x-y plane

    #print("radius = ", radius*1e3, "mm")
    print("x-y plane radius = ", format(np.max(XY)*1e3, '2f'), "mm")

    return Velocity, Position, Speed, Distance, XY



def fall_expansion(dt = 1e-4, g = 9.81, n = 10):
    """
    Evolution of atom cloud spatial and velocity distributions after MOT is turned off, due to gravity and thermal motion.
    Raman pulses not applie yet.
    """
    x = initial_cloud
    velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]

    Z = 0

    for j in range(0, n): # Motion of cloud before 1st pulse broken into "n" steps
        for i in range(0, N): # loop over all atoms
            
            #velocity[i][2] += g*dt # Change in atom speed along z-axis due to gravity

            position[i] += velocity[i]*dt # Displacement of atom after time dt
            
            Distance[i] = sqrt(sum(position[i][k]**2 for k in range(0, 3)))
            position_distribution = exp(-Distance[i]**2) # Gaussian distribution of atom distance
            Distance[i] *= position_distribution

            theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

            x, y, z = Distance[i]*sin(theta_r)*cos(phi_r), Distance[i]*sin(theta_r)*sin(phi_r), Distance[i]*cos(theta_r) # Position components

            position[i] = Vector(x, y, z) # Position vector

            XY[i] = sqrt(x**2 + y**2) # Distance of atom from cenetr of cloud in x-y plane

            Z += position[i][2]
        Z /= N # Mean distance by which atom cloud has fallen

        for i in range(0, N): # Subtract from "z" position coordinate mean distance fallen by atom cloud 
            position[i][2] -= Z
            Distance[i] = sqrt(sum(position[i][k]**2 for k in range(0, 3))) # Distance of atom from center of cloud
            
        radius = np.max(Distance) # Radius of atom cloud in x-y plane after time dt
            
        #print("radius = ", format(radius*1e3, '.2f'), "mm")
        print("x-y plane radius = ", format(np.max(XY)*1e3, '2f'), "mm")
        
    return velocity, position, Speed, Distance, XY


def rabi_frequency(time, r, I0_1, I0_2, w0_1, w0_2, delta, matrix_element, CG1, CG2, theta1 = 0, theta2 = 0, dt = 1e-4, g = 9.81):
    """
    Rabi frequency for one position. Have not taken into account broadening of Gaussian
    """
    step = time/dt
    x, y = r[step][0], r[step][1]
    I1 = I0_1*exp(-2*(((g*(time**2)/2)*sin(theta1) + x)**2 + y**2)/w0_1**2)
    I2 = I0_2*exp(-2*(((g*(time**2)/2)*sin(theta2) + x)**2 + y**2)/w0_2**2)
    rabi_eff = 2*sqrt(I1*I2)*(matrix_element**2)*CG1*CG2/(c*epsilon*(hbar**2)*2*delta)
    return rabi_eff


def light_atom(pulse, i = 1,  omega = 1e4, theta1 = 0, theta2 = 0):
    """
    Momentum recoil exerted on atom by two-photon transition
    Rabi oscillation determined by length of Raman pulse
        
    In this scenario excitation reduces atom speed, decay increases atom speed

    theta1, theta2 = angles of two laser beams relative to atom cloud trajectory
    omega = two-photon transition Rabi frequency (here, omega = 10 kHz)

    if i = -1: co-propagating beams
    if i = +1: counter-propagating beams

    if pulse = 1, 2 or 3: apply pulse 1, 2 or 3
    """
    w1, w2 = 384.2e12 + 4.271e9, 384.2e12 - 2.563e9 # Frequencies of beam photons
        
    k1, k2 = 2*pi*(w1/c), 2*pi*(w2/c) # wavevectors of beam photons

    # Effective wavevector. If i = -1, co-propagating beams: if i = +1, counter-propagating beams
    k_eff = k1*cos(theta1) + i*k2*cos(theta2) # z-axis momentum transfer
    k_y = k1*sin(theta1) + i*k2*sin(theta2) # xy plane momentum transfer

    dp = hbar*k_eff # Momentum recoil of atom after two-photon transition
    dv = dp/M # Change in z-axis speed due to atom-photon interactions
        
    dy = hbar*k_y/M # Change in speed in xy plane

    tau_a = pi/(2*omega) # π/2 pulse length
    tau_b = pi/omega # π pulse length

    if pulse == 1: # 1st π/2 pulse
        tau, d_tau, x = tau_a, tau_a, cloud_0
        velocity, position = x[0], x[1] # Atom velocities and positions before 1st pulse
        pulse_length = np.linspace(0, tau_a, 100) # 1st Raman pulse
        
    elif pulse == 2: # π pulse
        tau, d_tau, x = tau_b, tau_a + tau_b, interval_1
        velocity, position = x[0], x[1] # Atom velocities and positions before 2nd pulse
        pulse_length = np.linspace(tau_a, tau_a + tau_b, 100) # 2nd Raman pulse

    elif pulse == 3: # 2nd π/2 pulse
        tau, d_tau, x = tau_a, 2*tau_a + tau_b, interval_2
        velocity, position = x[0], x[1] # Atom velocities and positions before 3rd pulse
        pulse_length = np.linspace(tau_a + tau_b, 2*tau_a + tau_b, 100) # 3rd Raman pulse
        

    # Rabi oscilation of atom cloud between ground and excited states during Raman pulse
    Prob = np.sin(0.5*omega*pulse_length)**2
    Prob_final = sin(0.5*omega*d_tau)**2 # Probability of atom being in excited state at the end of a Raman pulse

    if pulse == 1: # Change in velocity and position of atom during 1st π/2 pulse

        for j in range(0, int(Prob_final*N)): # Atoms excited during pulse 1
            ###velocity[j][2] -= dv
            position[j][2] -= dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] -= dy*cos(phi)
                velocity[j][1] -= dy*sin(phi)

        
    elif pulse == 2: # Change in velocity and position of atom during π pulse

        for j in range(0, int(Prob_final*N)): # Atoms de-excited during pulse 2
            ###velocity[j][2] += dv
            position[j][2] += dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] += dy*cos(phi)
                velocity[j][1] += dy*sin(phi)


        for j in range(int(Prob_final*N) + 1, N): # Atoms excited during pulse 2
            ###velocity[j][2] -= dv
            position[j][2] -= dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] -= dy*cos(phi)
                velocity[j][1] -= dy*sin(phi)

    return velocity, position, Speed, Distance, XY, pulse_length, Prob


def interval(pulse, T = 1e-2, g = 9.81, n = 20):
    """
    Motion of atoms after 1st and 2nd Raman pulses
    Interval T between 2 consecutive pulses
    """
    Speed_1, Distance_1, Speed_2, Distance_2, XY1, XY2 = [], [], [], [], [], []

    print("\nInterval", "{}".format(pulse), ":")
        
    if pulse == 1: # Fall of atom cloud after 1st π/2 pulse
        x = pulse_1
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]
        print("Excited state atoms:", "\t\t", "Ground state atoms:")

    elif pulse == 2: # Fall of atom cloud after π pulse          
        x = pulse_2
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]
        print("Ground state atoms:", "\t\t", "Excited state atoms:")


    for j in range(0, n): # Motion of cloud between pulses broken into "n" steps
        for i in range(0, N):
            
        
        for i in range(0, int(0.5*N)): # Atoms excited by 1st π/2 pulse

            #velocity[i][2] += g*T # Change in atom speed along z-axis due to gravity

            #position[i] += velocity[i]*T # Displacement of atom after time dt
                    
            Distance_1.append(sqrt(sum(position[i][k]**2 for k in range(0, 3))))
            position_distribution = exp(-Distance_1[i]**2) # Gaussian distribution of atom distance
            Distance_1[i] *= position_distribution

            theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

            x, y, z = Distance_1[i]*sin(theta_r)*cos(phi_r), Distance_1[i]*sin(theta_r)*sin(phi_r), Distance_1[i]*cos(theta_r) # Position components

            position[i] = Vector(x, y, z) # Position vector

            Speed_1.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
            
        mean_speed_1 = np.mean(Speed_1) # Mean atom speed after time dt
        Temp_1 = ((pi*M)/(8*k))*mean_speed_1**2 # Temperature of atom cloud after time dt
        radius_1 = np.max(Distance_1) # Radius of atom cloud in x-y plane after time dt


        for i in range(int(0.5*N) + 1, N): # Atoms excited by 1st π/2 pulse

            m = int(i - 0.5*N - 1)

            #velocity[i][2] += g*T # Change in atom speed along z-axis due to gravity

            #position[i] += velocity[i]*T # Displacement of atom after time dt
                  
            Distance_2.append(sqrt(sum(position[i][k]**2 for k in range(0, 3))))
            position_distribution = exp(-Distance_2[m]**2) # Gaussian distribution of atom distance
            Distance_2[m] *= position_distribution

            theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

            x, y, z = Distance_2[m]*sin(theta_r)*cos(phi_r), Distance_2[m]*sin(theta_r)*sin(phi_r), Distance_2[m]*cos(theta_r) # Position components

            position[i] = Vector(x, y, z) # Position vector

            Speed_2.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
            
        mean_speed_2 = np.mean(Speed_2) # Mean atom speed after time dt
        Temp_2 = ((pi*M)/(8*k))*mean_speed_2**2 # Temperature of atom cloud after time dt
        radius_2 = np.max(Distance_2) # Radius of atom cloud in x-y plane after time dt

        print("radius =", format(radius_1*1e3, '.2f'), "mm", "\t\t", "radius =", format(radius_2*1e3, '.2f'), "mm")
        #print("mean speed =", format(mean_speed_1*1e3, '.2f'), "mm s-1", "\t", "mean speed =", format(mean_speed_2*1e3, '.2f'), "mm s-1")
        #print("Temperature =", format(Temp_1*1e6, '.2f'), "microK", "\t", "Temperature =", format(Temp_2*1e6, '.2f'), "microK")

    return velocity, position


def oscillation():
    """
    Plot of Rabi oscillations due to Raman pulses
    """
    x1, x2, x3 = pulse_1, pulse_2, pulse_3

    # Pulse length and probability evolution for each Raman pulse
    pulse_length_1, Prob_1 = x1[2], x1[3]
    pulse_length_2, Prob_2 = x2[2], x2[3]
    pulse_length_3, Prob_3 = x3[2], x3[3]

    # Plot of Rabi oscillations
    plt.figure(figsize = (6, 4))
    plt.plot(pulse_length_1, Prob_1, 'b', label = "pulse 1") # Oscillation after pulse 1
    plt.plot(pulse_length_2, Prob_2, 'r', label = "pulse 2") # Oscillation after pulse 2
    plt.plot(pulse_length_3, Prob_3, 'g', label = "pulse 3") # Oscillation after pulse 3
    plt.xlabel("time")
    plt.ylabel("Probability (time)")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc = 'upper right')
    plt.show()



if __name__ == "__main__":
    initial_cloud = initial_cloud() # Cloud at time t = 0

    cloud_0 = fall_expansion() # Cloud during initial fall

    pulse_1 = light_atom(1) # Cloud during 1st pulse
    interval_1 = interval(1) # Cloud during 1st interval
    #pulse_2 = light_atom(2) # Cloud during 2nd pulse
    #interval_2 = interval(2) # Cloud during 2nd interval
    #pulse_3 = light_atom(3) # Cloud during 3rd pulse

    #rabi = oscillation() # Variation of atom excitation probability
