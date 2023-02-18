from vp_sto.vpsto.vpsto import VPSTO
import numpy as np
from rl_planner import vpsto_wrapper, get_loss
import matplotlib.pyplot as plt


def loss_limits(candidates):
    q = candidates['pos']
    d_min = np.maximum(np.zeros_like(q), - q + q_min)
    d_max = np.maximum(np.zeros_like(q), q - q_max)
    return np.sum(d_min > 0.0, axis=(1,2)) + np.sum(d_max > 0.0, axis=(1,2))


def loss_curvature(candidates):
    dq = candidates['vel']
    ddq = candidates['acc']
    dq_sq = np.sum(dq**2, axis=-1)
    ddq_sq = np.sum(ddq**2, axis=-1)
    dq_ddq = np.sum(dq*ddq, axis=-1) 
    return np.mean((dq_sq * ddq_sq - dq_ddq**2) / (dq_sq**3 + 1e-6), axis=-1)

def loss(candidates):
    # print (candidates['pos'])
    # cost_curvature = loss_curvature(candidates)
    cost_limits = loss_limits(candidates)
    # return candidates['T'] + 1e-3 * cost_curvature + 1e-3 * cost_limits
    
    return vpsto_wrapper(candidates) + cost_limits
    # return cost_limits

vpsto = VPSTO(ndof=2)
q_min = np.array([0.276, -0.24]) # ([[0.226, 0.800], [-0.24, 0.24], [-0.0001, 0.4]])
q_max = np.array([0.724, 0.24])
vpsto.opt.vel_lim = np.array([0.2, 0.2]) # max. rad/s for each DoF
vpsto.opt.acc_lim = np.array([0.05, 0.05]) # max. rad/s^2 for each DoF
vpsto.opt.max_iter = 1 # 30 # max number of iterations
vpsto.opt.N_via = 3
vpsto.opt.N_eval = 7
vpsto.opt.pop_size = 2 # 4






q0 = np.array([0.35, -0.15]) # Current robot configuration
# qI = np.array([0.4, 0.05])
# qK = np.array([0.63, 0.02])
# qJ = np.array([0.5, 0.0])
qT = np.array([0.55, 0.15])  # Desired robot configuration
# joint_trajectory = np.vstack((np.linspace(q0,qI,3), np.linspace(qI,qJ,2), np.linspace(qJ,qK,2), np.linspace(qK,qT,3)))
# joint_trajectory = np.vstack((np.linspace(q0,qI,5), np.linspace(qI,qT,5)))
# joint_trajectory = np.linspace(q0,qT,10)
# print (joint_trajectory)
z = 0.1
# loss =  get_loss(joint_trajectory)
# print ("Final cost:",loss)
solution, via_points = vpsto.minimize(loss, q0, qT)
print (solution)
movement_duration = solution.T_best
t_traj = np.linspace(0, movement_duration, 1000)
joint_trajectory, vel, acc = solution.get_trajectory(t_traj)
print (f"Via points: {via_points}")
time_dt = (movement_duration/joint_trajectory.shape[0])
print ("Final point: ",joint_trajectory[-1])
print(f"Length of joint trajectory: {joint_trajectory.shape}")
plt.plot(solution.loss_list)
plt.show()

with open('paths/vpsto11.dat',mode='wb+') as f:
    for conf in joint_trajectory:
        conf = np.array([conf[0],conf[1],z])
        conf.tofile(f)

    final_cart = np.array([qT[0],qT[1],z])
    final_cart.tofile(f)
print ("Final cost:",min(solution.loss_list))

print(f"W best (best coeff. ): {solution.w_best}, {type(solution.w_best)}")
