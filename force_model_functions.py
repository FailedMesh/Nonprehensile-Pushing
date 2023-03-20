from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output



#########################################################################################################
# ------------------------------------------- HELPER FUNCTIONS ---------------------------------------- #
#########################################################################################################

def unpack_params(i, ext_force, contact_points, pos, vel):
    
    fcx, fcy = ext_force[i, :2]
    phi = contact_points[i, 0]

    theta = pos[i, 2]
    Rbs = get_rotation_matrix(theta)
    
    curr_vel = vel[i, :]
    
    return fcx, fcy, phi, Rbs, curr_vel

def unit_vector(v_input):
    
    v = v_input.copy()
    if np.linalg.norm(v) != 0.0:
        return v / np.linalg.norm(v)
    else:
        return v

def direction(v):
    
    if np.abs(v) != 0.0:
        return v / np.abs(v)
    else:
        return 0.0

def get_rotation_matrix(theta):

    return np.array([[np.cos(theta), np.sin(theta)], [- np.sin(theta), np.cos(theta)]])

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
    contact = np.round(np.array([r*np.cos(phi), r*np.sin(phi)]), 12)
    contact_unit = unit_vector(contact)
    
    force = (Rbs @ (np.array([[fcx, fcy]]).T)).T[0]
    force_unit = unit_vector(force)

    contact_angle = np.arctan2(force_unit[1], force_unit[0]) - np.arctan2(contact_unit[1], contact_unit[0])
    inverse_phi = np.arctan2(-contact_unit[1], -contact_unit[0])
    
    Fcom = np.dot(force, -contact_unit)
    
    Fbx = Fcom * np.cos(inverse_phi)
    Fby = Fcom * np.sin(inverse_phi)
    
    F_translational = np.round((Rbs.T @ (np.array([[Fbx, Fby]]).T)).T[0], 12)
    torque_force = np.round(np.linalg.norm(force) * np.sin(contact_angle), 12)
    
    vel_b = (Rbs @ (np.array([vel[:2]]).T)).T[0]
    inverse_vel_angle = np.arctan2(-vel_b[1], -vel_b[0])
    
    if np.linalg.norm(vel[:2]) != 0:
        Frx = Fr * np.cos(inverse_vel_angle)
        Fry = Fr * np.sin(inverse_vel_angle)
    elif abs(Fcom) > Fr:
        Frx = Fr * np.cos(phi)
        Fry = Fr * np.sin(phi)
    else:
        Frx = 0.0
        Fry = 0.0
    
    F_friction = np.round((Rbs.T @ (np.array([[Frx, Fry]]).T)).T[0], 12)
    
    if np.linalg.norm(vel[2]) != 0:
        torque_friction = -1 * Fr * direction(vel[2])
    elif abs(torque_force) > Fr:
        torque_friction = -1 * Fr * direction(torque_force)
    else:
        torque_friction = 0.0
    
    F = np.append(F_translational, torque_force)
    Friction = np.append(F_friction, torque_friction)
        
    #-----------Check if force is non-sliding--------------------#
    
    normal, tangent = get_contact_frame(phi, l, b)

    Fn = np.dot(normal, force)
    Ft = np.dot(tangent, force)

    if Fn < 0:
        raise Exception("Force is not directed towards the block")
    if np.abs(Ft) > mu2 * np.abs(Fn):
        raise Exception("Force applied will cause sliding against block surface, therefore it is invalid")
    
    #------------------------------------------------------------#
    
    F = F[:] + Friction[:]
    F[2] *= r
    
    return F


#########################################################################################################
# ----------------------------------------- GENERATE SIMULATION --------------------------------------- #
#########################################################################################################

def simulate(ext_force, phi, start_pos, start_vel, del_t, params):    

    fcx, fcy = ext_force
    Rbs = get_rotation_matrix(start_pos[2])

    F = transform_forces(fcx, fcy, phi, Rbs, start_vel, 
                         params['Fr'], params['mu2'], params['length'], params['breadth'])

    acc = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    pos = np.zeros((1, 3))
    
    cur_acc = np.divide(F, [params['mass'], params['mass'], params['Iz']])
    acc[0, :] = cur_acc.copy()
    vel[0, :] = start_vel.copy()
    pos[0, :] = start_pos.copy()

    #acc = np.concatenate([acc, cur_acc], axis = 0)

    i = 1
    #prev_fc = np.array([1., 1.])

    #for i in range(1, N):
    while True:
        
        clear_output(wait=True)
        print("Iterations = ", i)
        
        cur_vel = vel[i-1, :] + del_t * acc[i-1, :]
        cur_pos = pos[i-1, :] + del_t * vel[i-1, :]
        print(cur_pos)
        print(cur_vel)
        
        for j in range(3):
            
            if direction(cur_vel[j]) == -direction(vel[i-1, j]):
                cur_vel[j] = 0.0
        
        Rbs = get_rotation_matrix(cur_pos[2])
        F = transform_forces(0., 0., 0., Rbs, cur_vel, 
                         params['Fr'], params['mu2'], params['length'], params['breadth'])
        
        cur_acc = np.divide(F, [params['mass'], params['mass'], params['Iz']])

        acc = np.concatenate([acc, [cur_acc]], axis = 0)
        vel = np.concatenate([vel, [cur_vel]], axis = 0)
        pos = np.concatenate([pos, [cur_pos]], axis = 0)
        i += 1 

        #Stopping condition:
        if np.linalg.norm(cur_vel) == 0.0:
            break

        #prev_fc = np.array([fcx, fcy])

    return acc.copy(), vel.copy(), pos.copy()


def animate(pos, params):
    
    # create empty lists for the x and y data
    x = []
    y = []

    # create the figure and axes objects
    fig, ax = plt.subplots()

    length = params['length']
    breadth = params['breadth']

    # function that draws each frame of the animation
    def frame(i):

        x.append(pos[i, 0])
        y.append(pos[i, 1])

        ax.clear()
        ax.plot(x, y)
        block = plt.Rectangle((x[-1] - length/2, y[-1] - breadth/2), 
                                length, 
                                breadth, 
                                color = 'red', 
                                angle = pos[i, 2], 
                                rotation_point = (x[-1], y[-1]))
        plt.gca().add_patch(block)

        upper_limit = np.max(pos[:, :2]) + 2*length
        lower_limit = np.min(pos[:, :2]) - 2*length

        ax.set_xlim([lower_limit, upper_limit])
        ax.set_ylim([lower_limit, upper_limit])

    # run the animation
    ani = FuncAnimation(fig, frame, frames = pos.shape[0], interval=20, repeat=False)

    plt.show()