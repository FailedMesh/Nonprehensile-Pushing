from random import randint
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#########################################################################################################
# ---------------------------------------- ENVIRONMENT PARAMETERS ------------------------------------- #
#########################################################################################################

#Time-step
del_t = 0.01

#Friction coefficient between slider and surface:
mu1 = 0.3

#Friction coefficient between gripper and slider:
mu2 = 0.1

#Block dimensions:
length = 0.3
breadth = 0.1
height = 0.1

#Mass of block in kg:
mass = 0.2

#Moment of Inertia (Cuboid here):
Iz = (mass * (length**2 + breadth**2)) / 12

#Friction:
Fr = mu1 * mass * 9.81

#Start Point (x, y, theta):
start = np.array([0.3, 0.3, 0.0])

#########################################################################################################
# ------------------------------------------- HELPER FUNCTIONS ---------------------------------------- #
#########################################################################################################

def unpack_params(i, ext_force, contact_points, pos, vel):
    
    fcx, fcy = ext_force[i, :2]
    phi = contact_points[i, 0]

    theta = pos[i, 2]
    Rbs = np.array([[np.cos(theta), np.sin(theta)],
                    [- np.sin(theta), np.cos(theta)]])
    
    curr_vel = vel[i, :]
    
    return fcx, fcy, phi, Rbs, curr_vel

def unit_vector(v):
    
    if np.linalg.norm(v) != 0.0:
        return v / np.linalg.norm(v)
    else:
        return v

def direction(v):
    
    if np.abs(v) != 0.0:
        return v / np.abs(v)
    else:
        return 0.0

#########################################################################################################
# ------------------------------------------ GEOMETRY FUNCTIONS --------------------------------------- #
#########################################################################################################

def slider_geometry(phi, l, b):
    
    # phi belongs to [-pi, pi]
    
    if phi > np.pi:
        phi = phi - 2*np.pi
    elif phi < -np.pi:
        phi = 2*np.pi + phi
    
    interval1 = - np.arctan(b/l) - 2*np.arctan(l/b)
    interval2 = - np.arctan(b/l)
    interval3 = np.arctan(b/l)
    interval4 = np.arctan(b/l) + 2*np.arctan(l/b)
    
    if (phi > interval1 and phi < interval2) or (phi > interval3 and phi < interval4):
        r = b/(2*np.sin(phi))
        
    else:
        r = l/(2*np.cos(phi))
        
    return np.abs(r)

def get_contact_frame(phi, l, b):
    
    r = slider_geometry(phi, l, b)
    contact = np.array([r*np.cos(phi), r*np.sin(phi)])
    contact_unit = unit_vector(contact)
    
    r_front = slider_geometry(phi + 0.001, l, b)
    contact_front = np.array([r_front*np.cos(phi + 0.001), r_front*np.sin(phi + 0.001)])
    
    r_back = slider_geometry(phi - 0.001, l, b)
    contact_back = np.array([r_back*np.cos(phi - 0.001), r_back*np.sin(phi - 0.001)])
    
    front_dir = unit_vector(contact_front - contact)
    back_dir = unit_vector(contact_back - contact)
    
    front_angle = np.arctan2(front_dir[1], front_dir[0])
    back_angle = np.arctan2(back_dir[1], back_dir[0])
    
    angle_bisect = (back_angle - front_angle) / 2
    angle_bisect = angle_bisect + (1 - direction(angle_bisect)) * np.pi / 2
    
    normal_angle = front_angle + angle_bisect
    tangent_angle = normal_angle - np.pi/2
    
    normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
    tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
    
    return normal, tangent

def transform_forces(fcx, fcy, phi, Rbs, vel, Fr, mu2, l, b):
    
    r = slider_geometry(phi, l, b)
    contact = np.array([r*np.cos(phi), r*np.sin(phi)])
    contact_unit = unit_vector(contact)
    
    force = (Rbs @ (np.array([[fcx, fcy]]).T)).T[0]
    force_unit = unit_vector(force)
    
    contact_angle = np.arctan2(force_unit[1], force_unit[0]) - np.arctan2(contact_unit[1], contact_unit[0])
    inverse_phi = np.arctan2(-contact_unit[1], -contact_unit[0])
    
    Fcom = np.dot(force, -contact_unit)
    
    Fbx = Fcom * np.cos(inverse_phi)
    Fby = Fcom * np.sin(inverse_phi)
    
    F_translational = (Rbs.T @ (np.array([[Fbx, Fby]]).T)).T[0]
    torque_force = np.linalg.norm(force) * np.sin(contact_angle)
    
    F = np.append(F_translational, torque_force)
    
    #Check if force is non-sliding:
    
    normal, tangent = get_contact_frame(phi, l, b)

    Fn = np.dot(normal, force)
    Ft = np.dot(tangent, force)

    if Fn < 0:
        raise Exception("Force is not directed towards the block")
    if np.abs(Ft) > mu2 * np.abs(Fn):
        raise Exception("Force applied will cause sliding against block surface, therefore it is invalid")
    
    #Accounting for Friction:
    
    for i in range(F.size):
        
        if (vel[i] != 0):
            F[i] -= Fr * direction(vel[i])
        elif (vel[i] == 0 and abs(F[i]) > Fr):
            F[i] -= Fr * direction(F[i])
        else:
            F[i] = 0.0
        
    F[2] *= r
    
    return F

#########################################################################################################
# ----------------------------------------------- PHYSICS --------------------------------------------- #
#########################################################################################################

# ************* System States ************* #

#Total simulation time in seconds:
T = 5

#Total number of trajectory points:
N = int(T/del_t + 1)

#Initialize arrays for states
pos = np.zeros((N, 3))
vel = np.zeros((N, 3))
acc = np.zeros((N, 3))

#Assign the initial position
pos[0, :] = start[:]

# ************* External Parameters ************* #

ext_force = np.zeros((N, 2))
#contact_points = np.zeros((N, 1))
contact_points = np.ones((N, 1)) * (-20 * np.pi/180)

#Initial force:
ext_force[0, 0] = 0.0
ext_force[0, 1] = 60.0

#########################################################################################################
# ----------------------------------------- GENERATE SIMULATION --------------------------------------- #
#########################################################################################################

fcx, fcy, phi, Rbs, curr_vel = unpack_params(0, ext_force, contact_points, pos, vel)
F = transform_forces(fcx, fcy, phi, Rbs, curr_vel, Fr, mu2, length, breadth)

acc[0, :2] = F[:2] / mass
acc[0, 2] = F[2] / Iz

prev_fcx = 1.0
prev_fcy = 1.0

for i in range(1, N):
    
    vel[i, :] = vel[i-1, :] + del_t * acc[i-1, :]
    pos[i, :] = pos[i-1, :] + del_t * vel[i-1, :]
    
    fcx, fcy, phi, Rbs, curr_vel = unpack_params(i, ext_force, contact_points, pos, vel)
    F = transform_forces(fcx, fcy, phi, Rbs, curr_vel, Fr, mu2, length, breadth)
    
    acc[i, :2] = F[:2] / mass
    acc[i, 2] = F[2] / Iz
    
    if np.linalg.norm(prev_fcx) == 0 and np.linalg.norm(prev_fcy) == 0 and np.dot(vel[i, :], vel[i-1, :]) < 0:
        
        vel[i, :] = np.zeros((3, ))
        
    prev_fcx = fcx
    prev_fcy = fcy

#########################################################################################################
# ---------------------------------------- VISUALIZE SIMULATION --------------------------------------- #
#########################################################################################################

# create empty lists for the x and y data
x = []
y = []

# create the figure and axes objects
fig, ax = plt.subplots()

# function that draws each frame of the animation
def animate(i):
    
    x.append(pos[i, 0])
    y.append(pos[i, 1])

    ax.clear()
    ax.plot(x, y)
    # block = plt.Rectangle((x[-1] - length/2, y[-1] - breadth/2), 
    #                         length, 
    #                         breadth, 
    #                         color = 'red', 
    #                         angle = pos[i, 2], 
    #                         rotation_point = (x[-1], y[-1]))
    #plt.gca().add_patch(block)
    ax.set_xlim([np.min(pos[:, 0]) - 2*length, np.max(pos[:, 0]) + 2*length])
    ax.set_ylim([np.min(pos[:, 1]) - 2*length, np.max(pos[:, 1]) + 2*length])

# run the animation
ani = FuncAnimation(fig, animate, frames=N, interval=20, repeat=False)

plt.show()

_ = input("Press Enter to close")
