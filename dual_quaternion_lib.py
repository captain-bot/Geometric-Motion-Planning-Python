import numpy as np
import math


# Given two quaternions find quaternion product using matrix multiplication rule
def quat_prod(q1, q2):
    q1_left = np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                        [q1[1], q1[0], -q1[3], q1[2]],
                        [q1[2], q1[3], q1[0], -q1[1]],
                        [q1[3], -q1[2], q1[1], q1[0]]])
    return np.dot(q1_left, q2)


# Given two quaternions find quaternion product using quaternion multiplication rule
def quat_prod_basic(q1, q2):
    return np.hstack((q1[0]*q2[0] - np.dot(q1[1:], q2[1:]), q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])))


# Given a quaternion return conjugate of that quaternion
def conjugate_quat(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# Given a dual-quaternion return its conjugate
def conjugate_dual_quat(dq):
    real_part = conjugate_quat(dq[0, :].copy())
    dual_part = - conjugate_quat(dq[1, :].copy())
    return np.vstack((real_part, dual_part))


# Given a rigid transformation return dual quaternion
def rigid_trans_2_dual_quat(g_in):
    alpha = rotm2quat(g_in[:3, :3].copy())
    if alpha is None:
        return None
    position_quat = np.array([0, g_in[0, 3], g_in[1, 3], g_in[2, 3]])
    beta = 0.5 * quat_prod(position_quat, alpha)
    return np.vstack((alpha, beta))


# Given dual quaternion return homogeneous transformation matrix
def dual_quat_2_homomat(d_quat):
    rotmat = unitquat2rotm(d_quat[0, :])
    p_vec = 2 * quat_prod(d_quat[1, :], conjugate_quat(d_quat[0, :]))
    print("p_vec: ")
    print(p_vec)
    print("Rotation matrix: ")
    print(rotmat)
    return p_vec, rotmat


# Given two dual-quaternions compute their product
def dual_quat_prod(dq1, dq2):
    p, q = dq1[0, :], dq1[1, :]
    u, v = dq2[0, :], dq2[1, :]
    real_part = dual_quat_prod(p, u)
    dual_part = dual_quat_prod(q, u) + dual_quat_prod(p, v)
    return np.vstack((real_part, dual_quat))


# Power of a given dual-quaternion
def dual_quat_pow(dq):
    real_part = dq[0, :]
    dual_part = dq[1, :]



# Quaternion to rotation matrix
def unitquat2rotm(q):
    return np.array([[math.pow(q[0], 2) + math.pow(q[1], 2) - math.pow(q[2], 2) - math.pow(q[3], 2),
            2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
            [2 * q[1] * q[2] + 2 * q[0] * q[3],
            math.pow(q[0], 2) - math.pow(q[1], 2) + math.pow(q[2], 2) - math.pow(q[3], 2),
            2 * q[2] * q[3] - 2 * q[0] * q[1]],
            [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1],
            math.pow(q[0], 2) - math.pow(q[1], 2) - math.pow(q[2], 2) + math.pow(q[3], 2)]])


# Rotation matrix to quaternion
def rotm2quat(rotm):
    if (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 < -1 or (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 > 1:
        return None
    angle = math.acos((rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2)
    # print(angle)
    temp = math.sqrt((rotm[2, 1] - rotm[1, 2])**2 + (rotm[0, 2] - rotm[2, 0])**2 + (rotm[1, 0] - rotm[0, 1])**2)
    wx = (rotm[2, 1] - rotm[1, 2]) / temp
    wy = (rotm[0, 2] - rotm[2, 0]) / temp
    wz = (rotm[1, 0] - rotm[0, 1]) / temp
    quat = np.array([math.cos(angle / 2), wx * math.sin(angle / 2), wy * math.sin(angle / 2), wz * math.sin(angle / 2)])
    return quat


# Input quaternion
quat1 = np.array([0.1, 0.2, 0.5, 0.1])
quat2 = np.array([0.2, 0.3, -0.2, 0.1])

print(quat_prod(quat1, quat2))
print(quat_prod_basic(quat1, quat2))


# Input rigid transformation
# g = np.array([[0.707, -0.707, 0, 5.6], [0.707, 0.707, 0, 0.5], [0, 0, 1, 800.5], [0, 0, 0, 1]])
g = np.array([[1, 0, 0, 5.6], [0, 0.707, -0.707, 0.5], [0, 0.707, 0.707, 800.5], [0, 0, 0, 1]])
dual_quat = rigid_trans_2_dual_quat(g)
if not dual_quat is None:
    print(dual_quat)
else:
    print("Dual quaternion can not be obtained")


# Input given a dual quaternion return homogeneous transformation matrix
g_back = dual_quat_2_homomat(dual_quat)



