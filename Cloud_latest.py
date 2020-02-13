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
    Velocity, Position, Speed, Distance, XY = [], [], [], [], []

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

        XY.append(sqrt(x**2 + y**2)) # Distance of atom from center of cloud in x-y plane

    XY_max = np.max(XY)

    #print("radius = ", radius*1e3, "mm")
    print("x-y radius = ", format(XY_max*1e3, '2f'), "mm")

    return Velocity, Position, Speed, Distance, XY



def fall_expansion(dt = 1e-4, g = 9.81, n = 20):
    """
    Evolution of atom cloud spatial and velocity distributions after MOT is turned off, due to gravity and thermal motion.
    Raman pulses not applie yet.
    """
    x = initial_cloud
    velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]

    for j in range(0, n): # Motion of cloud before 1st pulse broken into "n" steps
        for i in range(0, N): # loop over all atoms
            
            #velocity[i][2] += g*dt # Change in atom speed along z-axis due to gravity
            #Speed[i] = sqrt(sum(velocity[i][k]**2 for k in range(0, 3)))

            position[i] += velocity[i]*dt # Displacement of atom after time dt
            
            Distance[i] = sqrt(sum(position[i][k]**2 for k in range(0, 3)))
            position_distribution = exp(-Distance[i]**2) # Gaussian distribution of atom distance from initial cloud center
            Distance[i] *= position_distribution

            theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

            x, y, z = Distance[i]*sin(theta_r)*cos(phi_r), Distance[i]*sin(theta_r)*sin(phi_r), Distance[i]*cos(theta_r) # Position components

            position[i] = Vector(x, y, z) # Position vector

            XY[i] = sqrt(x**2 + y**2) # Distance of atom from cenetr of cloud in x-y plane
            
        radius = np.max(Distance) # Radius of atom cloud in x-y plane after time dt
            
        #print("radius = ", format(radius*1e3, '.2f'), "mm")
        print("x-y radius = ", format(np.max(XY)*1e3, '2f'), "mm")
        
    return velocity, position, Speed, Distance, XY


def rabi_frequency(time, r, I0_1, I0_2, w0_1, w0_2, delta, matrix_element, CG1, CG2, theta1 = 0, theta2 = 0, dt = 1e-4, g = 9.81):
    """
    Calculation of position-dependent Rabi frequency.

    Initial conditions:
    1) Both beams are parallel and coaligned with atom clodu trajectory.
    2) Both beams are perfectly collimated.
    2) 2-photon transition is resonant.
    3) No further detuning required due to Doppler effect.
    4) Only 1 path for 2-photon transition, so only 1 pair of CG coefficients.
    """
    step = time/dt # Time step
    
    x, y = r[step][0], r[step][1] # x, y coordinates where beam intensities are computed
    
    I1 = I0_1*exp(-2*(((g*(time**2)/2)*sin(theta1) + x)**2 + y**2)/w0_1**2) # Intensity of beam 1
    
    I2 = I0_2*exp(-2*(((g*(time**2)/2)*sin(theta2) + x)**2 + y**2)/w0_2**2) # Intensity of beam 2
    
    rabi_eff = 2*sqrt(I1*I2)*(matrix_element**2)*CG1*CG2/(c*epsilon*(hbar**2)*2*delta) # Rabi frequency at x, y position
    
    return rabi_eff


def light_atom(pulse, i = 1, theta1 = 0, theta2 = 0):
    """
    Momentum recoil exerted on atom by two-photon transition
    Rabi oscillation determined by length of Raman pulse
        
    In this scenario excitation reduces atom speed, decay increases atom speed

    theta1, theta2 = angles of two laser beams relative to atom cloud trajectory

    if i = -1: co-propagating beams
    if i = +1: counter-propagating beams

    if pulse = 1, 2 or 3: apply pulse 1, 2 or 3
    """
    
    frequency = omega

    w1, w2 = 384.2e12 + 4.271e9, 384.2e12 - 2.563e9 # Frequencies of beam photons
        
    k1, k2 = 2*pi*(w1/c), 2*pi*(w2/c) # wavevectors of beam photons

    # Effective wavevector. If i = -1, co-propagating beams: if i = +1, counter-propagating beams
    k_eff = k1*cos(theta1) + i*k2*cos(theta2) # z-axis momentum transfer
    k_y = k1*sin(theta1) + i*k2*sin(theta2) # xy plane momentum transfer

    dp = hbar*k_eff # Momentum recoil of atom after two-photon transition
    dv = dp/M # Change in z-axis speed due to atom-photon interactions
        
    dy = hbar*k_y/M # Change in speed in xy plane

    tau_a = pi/(2*frequency) # ?/2 pulse length
    tau_b = pi/frequency # ? pulse length

    # For each pulse, several atom-specific parameters are defined:
    # velocity, position, speed, distance from initial cloud center, distance from cloud center in x-y plane
    if pulse == 1: # 1st ?/2 pulse
        tau, d_tau, x = tau_a, tau_a, cloud_0
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4] # Atom velocities and positions before 1st pulse
        pulse_length = np.linspace(0, tau_a, 100) # 1st Raman pulse
        
    elif pulse == 2: # ? pulse
        tau, d_tau, x = tau_b, tau_a + tau_b, interval_1
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4] # Atom velocities and positions before 2nd pulse
        pulse_length = np.linspace(tau_a, tau_a + tau_b, 100) # 2nd Raman pulse

    elif pulse == 3: # 2nd ?/2 pulse
        tau, d_tau, x = tau_a, 2*tau_a + tau_b, interval_2
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4] # Atom velocities and positions before 3rd pulse
        pulse_length = np.linspace(tau_a + tau_b, 2*tau_a + tau_b, 100) # 3rd Raman pulse
        

    # Rabi oscilation of atom cloud between ground and excited states during Raman pulse
    Prob = np.sin(0.5*frequency*pulse_length)**2
    Prob_final = sin(0.5*frequency*d_tau)**2 # Probability of atom being in excited state at the end of a Raman pulse

    if pulse == 1: # Change in velocity and position of atom during 1st ?/2 pulse

        for j in range(0, int(Prob_final*N)): # Atoms excited during pulse 1
            ###velocity[j][2] -= dv
            position[j][2] -= dv*tau

            if theta1 or theta2 != 0:
                phi = 2*pi*random.random()
                velocity[j][0] -= dy*cos(phi)
                velocity[j][1] -= dy*sin(phi)

        
    elif pulse == 2: # Change in velocity and position of atom during ? pulse

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
    Velocity_1, Position_1, Speed_1, Distance_1, XY_1 = [], [], [], [], []
    Velocity_2, Position_2, Speed_2, Distance_2, XY_2 = [], [], [], [], []

    print("\nInterval", "{}".format(pulse), ":")
        
    if pulse == 1: # Fall of atom cloud after 1st ?/2 pulse
        x = pulse_1
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]
        print("Excited state atoms:", "\t\t", "Ground state atoms:")

    elif pulse == 2: # Fall of atom cloud after ? pulse          
        x = pulse_2
        velocity, position, Speed, Distance, XY = x[0], x[1], x[2], x[3], x[4]
        print("Ground state atoms:", "\t\t", "Excited state atoms:")


    for j in range(0, n): # Motion of cloud between pulses broken into "n" steps
        
        for i in range(0, N):

            #velocity[i][2] += g*T # Change in atom speed along z-axis due to gravity
            #Speed[i] = sqrt(sum(velocity[i][k]**2 for k in range(0, 3)))

            #position[i] += velocity[i]*T # Displacement of atom after time dt

            Distance[i] = sqrt(sum(position[i][k]**2 for k in range(0, 3)))
            position_distribution = exp(-Distance[i]**2) # Gaussian distribution of atom distance
            Distance[i] *= position_distribution

            theta_r, phi_r = pi*random.uniform(-1, 1), 2*pi*random.random() # Polar and azimuthal angles of atom position vector

            x, y, z = Distance[i]*sin(theta_r)*cos(phi_r), Distance[i]*sin(theta_r)*sin(phi_r), Distance[i]*cos(theta_r) # Position components

            position[i] = Vector(x, y, z) # Position vector

            XY[i] = sqrt(x**2 + y**2) # Distance of atom from cenetr of cloud in x-y plane

            
        for i in range(0, int(0.5*N)): # Atoms excited by 1st ?/2 pulse
            
            Velocity_1.append(velocity[i])
            Position_1.append(position[i])
            Speed_1.append(Speed[i])
            Distance_1.append(Distance[i])
            XY_1.append(XY[i])
            
            
        mean_speed_1 = np.mean(Speed_1) # Mean atom speed after time dt
        Temp_1 = ((pi*M)/(8*k))*mean_speed_1**2 # Temperature of atom cloud after time dt
        radius_1 = np.max(Distance_1) # Radius of atom cloud in x-y plane after time dt


        for i in range(int(0.5*N) + 1, N): # Atoms excited by 1st ?/2 pulse

            Velocity_2.append(velocity[i])
            Position_2.append(position[i])
            Speed_2.append(Speed[i])
            Distance_2.append(Distance[i])
            XY_2.append(XY[i])

            
        mean_speed_2 = np.mean(Speed_2) # Mean atom speed after time dt
        Temp_2 = ((pi*M)/(8*k))*mean_speed_2**2 # Temperature of atom cloud after time dt
        radius_2 = np.max(Distance_2) # Radius of atom cloud in x-y plane after time dt

        #print("radius =", format(radius_1*1e3, '.2f'), "mm", "\t\t", "radius =", format(radius_2*1e3, '.2f'), "mm")
        #print("mean speed =", format(mean_speed_1*1e3, '.2f'), "mm s-1",
        #      "\t", "mean speed =", format(mean_speed_2*1e3, '.2f'), "mm s-1")
        #print("Temperature =", format(Temp_1*1e6, '.2f'), "microK", "\t", "Temperature =", format(Temp_2*1e6, '.2f'), "microK")
        print("x-y radius 1 = ", format(np.max(XY_1)*1e3, '1f'), "mm", "\t",
              "x-y radius 2 = ", format(np.max(XY_2)*1e3, '1f'), "mm")

    return velocity, position, Speed, Distance, XY


def oscillation():
    """
    Plot of Rabi oscillations due to Raman pulses
    """
    x1, x2, x3 = pulse_1, pulse_2, pulse_3

    # Pulse length and probability evolution for each Raman pulse
    pulse_length_1, Prob_1 = x1[5], x1[6]
    pulse_length_2, Prob_2 = x2[5], x2[6]
    pulse_length_3, Prob_3 = x3[5], x3[6]

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

    #omega = rabi_frequency()

    pulse_1 = light_atom(1) # Cloud during 1st pulse
    interval_1 = interval(1) # Cloud during 1st interval
    pulse_2 = light_atom(2) # Cloud during 2nd pulse
    interval_2 = interval(2) # Cloud during 2nd interval
    pulse_3 = light_atom(3) # Cloud during 3rd pulse

    rabi = oscillation() # Variation of atom excitation probability
