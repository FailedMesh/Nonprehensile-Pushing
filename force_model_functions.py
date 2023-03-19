from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



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
        
    F = F[:] + Friction[:]
    F[2] *= r
    
    return F
