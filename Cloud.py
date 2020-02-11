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

def initial_cloud(Temp_0 = 1e-5, diameter = 1e-3):
    """
    Initial conditions of the atom cloud: i.e. at time t = 0 when the MOT is turned off.
    Assumptions: no interatomic collisions, uniform cloud.
        
    Initial radius, spatial and velocity distributions and temperature.
    """
    radius = diameter/2 # Cloud radius at time t = 0
        
    # Speed, velocity and spatial distributions
    V, R, Mean_V = [], [], []
    for i in range(0, N):
            
        v = random.uniform(1e-5, 1) # Atom speed in MOT
        Prob = ((M/(2*pi*k*Temp_0))**(3/2))*(4*pi*v**2)*exp(-(M*v**2)/(2*k*Temp_0)) # Boltzmann distribution function of atom speed
        v *= Prob # Weighted atom speed
            
        theta, phi = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles
            
        vx, vy, vz = v*sin(theta)*cos(phi), v*sin(theta)*sin(phi), v*cos(theta) # Velocity components
        velocity = Vector(vx, vy, vz) # Velocity vector
        V.append(velocity)

        Mean_V.append(sqrt(sum(velocity[j]**2 for j in range(0, 3))))

        x, y, z = radius*random.uniform(-1, 1), radius*random.uniform(-1, 1), radius*random.uniform(-1, 1) # Position components
        position = Vector(x, y, z) # Position vector
        R.append(position)

    print("Temperature = ", Temp_0, "K")
    print("radius = ", format(radius*1e3, '.2f'), "mm")
    print("mean speed = ", format(np.mean(Mean_V)*1e3, '.2f'), "mm s-1")

    return V, R


def fall_expansion(dv = 1e-3, dt = 1e-4, g = 9.81, n = 10):
    """
    Evolution of atom cloud spatial and velocity distributions after MOT is turned off, due to gravity and thermal motion.
    Raman pulses not applie yet.
    """
    x = initial_cloud
    velocity, position = x[0], x[1]
    Mean_V, Max_R = [], []

    for j in range(0, n): # Motion of cloud before 1st pulse broken into "n" steps
        for i in range(0, int(N)): # loop over all atoms
            position[i] += velocity[i]*dt # Displacement of atom after time dt
            velocity[i] += (dv*Vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))) # Change in atom velocity caused by thermal motion
            velocity[i][2] += g*dt # Change in atom speed along z-axis due to gravity
                    
            Mean_V.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
            Max_R.append(sqrt(position[i][0]**2 + position[i][1]**2))
            
        mean_speed = np.mean(Mean_V) # Mean atom speed after time dt
        Temp = ((pi*M)/(8*k))*mean_speed**2 # Temperature of atom cloud after time dt
        radius = np.max(Max_R) # Radius of atom cloud in x-y plane after time dt
            
        print("radius = ", format(radius*1e3, '.2f'), "mm")
        print("mean speed = ", format(mean_speed*1e3, '.2f'), "mm s-1")
        print("Temperature =", format(Temp*1e6, '.2f'), "microK")
        
    return velocity, position


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
        tau, x = tau_a, cloud_0
        velocity, position = x[0], x[1] # Atom velocities and positions before 1st pulse
        pulse_length = np.linspace(0, tau_a, 100) # 1st Raman pulse
        
    elif pulse == 2: # π pulse
        tau, x = tau_b, interval_1
        velocity, position = x[0], x[1] # Atom velocities and positions before 2nd pulse      
        pulse_length = np.linspace(tau_a, tau_a + tau_b, 100) # 2nd Raman pulse

    elif pulse == 3: # 2nd π/2 pulse
        tau, x = tau_a, interval_2
        velocity, position = x[0], x[1] # Atom velocities and positions before 3rd pulse
        pulse_length = np.linspace(tau_a + tau_b, 2*tau_a + tau_b, 100) # 3rd Raman pulse
        

    # Rabi oscilation of atom cloud between ground and excited states during Raman pulse
    Prob = np.sin(0.5*omega*pulse_length)**2
    Prob_final = np.sin(0.5*omega*tau)**2 # Probability of atom being in excited state at the end of a Raman pulse

    if pulse == 1: # Change in velocity and position of atom during 1st π/2 pulse
        for j in range(0, int(Prob_final*N)): # Atoms excited during pulse 1
            velocity[j][2] -= dv
            position[j][2] -= dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] -= dy*cos(phi)
                velocity[j][1] -= dy*sin(phi)

        
    elif pulse == 2: # Change in velocity and position of atom during π pulse

        for j in range(0, int(Prob_final*N)): # Atoms de-excited during pulse 2
            velocity[j][2] += dv
            position[j][2] += dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] += dy*cos(phi)
                velocity[j][1] += dy*sin(phi)


        for j in range(int(Prob_final*N) + 1, N): # Atoms excited during pulse 2
            velocity[j][2] -= dv
            position[j][2] -= dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] -= dy*cos(phi)
                velocity[j][1] -= dy*sin(phi)

    return velocity, position, pulse_length, Prob


def interval(pulse, T = 1e-2, dv = 1e-3, g = 9.81, n = 10):
    """
    Motion of atoms after 1st and 2nd Raman pulses
    Interval T between 2 consecutive pulses
    """
    Mean_V1, Max_R1, Mean_V2, Max_R2 = [], [], [], []

    print("\nInterval", "{}".format(pulse), ":")
        
    if pulse == 1: # Fall of atom cloud after 1st π/2 pulse
        x = pulse_1
        velocity, position = x[0], x[1]
        print("Excited state atoms:", "\t\t", "Ground state atoms:")

    elif pulse == 2: # Fall of atom cloud after π pulse          
        x = pulse_2
        velocity, position = x[0], x[1]
        print("Ground state atoms:", "\t\t", "Excited state atoms:")


    for j in range(0, n): # Motion of cloud between pulses broken into "n" steps
        for i in range(0, int(0.5*N)): # Atoms excited by 1st π/2 pulse

            position[i] += velocity[i]*(T/10) # Displacement of atom after time dt
            velocity[i] += dv*Vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) # Change in atom velocity caused by thermal motion
            velocity[i][2] += g*(T/n) # Change in atom speed along z-axis due to gravity
                    
            Mean_V1.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
            Max_R1.append(sqrt(position[i][0]**2 + position[i][1]**2))
            
        mean_speed_1 = np.mean(Mean_V1) # Mean atom speed after time dt
        Temp_1 = ((pi*M)/(8*k))*mean_speed_1**2 # Temperature of atom cloud after time dt
        radius_1 = np.max(Max_R1) # Radius of atom cloud in x-y plane after time dt


        for i in range(int(0.5*N) + 1, N):

            position[i] += velocity[i]*(T/10) # Displacement of atom after time dt
            velocity[i] += dv*Vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) # Change in atom velocity caused by thermal motion
            velocity[i][2] += g*(T/n) # Change in atom speed along z-axis due to gravity
                  
            Mean_V2.append(sqrt(sum(velocity[i][k]**2 for k in range(0, 3))))
            Max_R2.append(sqrt(position[i][0]**2 + position[i][1]**2))
            
        mean_speed_2 = np.mean(Mean_V2) # Mean atom speed after time dt
        Temp_2 = ((pi*M)/(8*k))*mean_speed_2**2 # Temperature of atom cloud after time dt
        radius_2 = np.max(Max_R2) # Radius of atom cloud in x-y plane after time dt

        print("radius =", format(radius_1*1e3, '.2f'), "mm", "\t\t", "radius =", format(radius_2*1e3, '.2f'), "mm")
        print("mean speed =", format(mean_speed_1*1e3, '.2f'), "mm s-1", "\t", "mean speed =", format(mean_speed_2*1e3, '.2f'), "mm s-1")
        print("Temperature =", format(Temp_1*1e6, '.2f'), "microK", "\t", "Temperature =", format(Temp_2*1e6, '.2f'), "microK")

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




initial_cloud = initial_cloud() # Cloud at time t = 0

cloud_0 = fall_expansion() # Cloud during initial fall

pulse_1 = light_atom(1) # Cloud during 1st pulse
interval_1 = interval(1) # Cloud during 1st interval
pulse_2 = light_atom(2) # Cloud during 2nd pulse
interval_2 = interval(2) # Cloud during 2nd interval
pulse_3 = light_atom(3) # Cloud during 3rd pulse

rabi = oscillation() # Variation of atom excitation probability
