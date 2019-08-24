import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This file contains three main classes namely "Quaternion", "Dual-Quaternion" and
"IntpRigidDispDualQuat".

The class "IntpRigidDispDualQuat" can be used to obtain smooth interpolation 
betweeen two rigid displacements. The constructor of this class takes 4 input
arguents all in numpy.array format. The first two are position and rotation 
quaternion of initial pose whereas the last two are position and quayernion 
vectors of final pose.

Author: Anirban Sinha
Last updated: August 21st., 2019
"""


# Quaternion class
class Quaternion(object):
    def __init__(self, q):
        self.scalar = q[0]
        self.vector = q[1:]
        self.total = q

    # Given two quaternions find quaternion product using matrix multiplication rule
    def quat_prod(self, q2):
        q1_left = np.array([[self.total[0], -self.total[1], -self.total[2], -self.total[3]],
                            [self.total[1], self.total[0], -self.total[3], self.total[2]],
                            [self.total[2], self.total[3], self.total[0], -self.total[1]],
                            [self.total[3], -self.total[2], self.total[1], self.total[0]]])
        return Quaternion(np.dot(q1_left, q2.total))

    # Given two quaternions find quaternion product using quaternion multiplication rule
    def quat_prod_basic(self, q2):
        return Quaternion(np.hstack(
            (self.total[0] * q2.total[0] - np.dot(self.total[1:], q2.total[1:]), self.total[0] * q2.total[1:]
             + q2.total[0] * self.total[1:] + np.cross(self.total[1:], q2.total[1:]))))

    # Given a quaternion return conjugate of that quaternion
    def conjugate_quat(self):
        return Quaternion(np.array([self.total[0], -self.total[1], -self.total[2], -self.total[3]]))

    # Get Axis-Angle representation of rotation matrix
    def get_axis_angle(self):
        new_q = self.total/np.dot(self.total, self.total)
        # print(new_q)
        if new_q[0] == 1:
            # raise Exception("scalar part can not be zero\n")
            ang = 0
            ax = np.array([1, 0, 0])
        else:
            ang = 2 * np.arccos(new_q[0])
            ax = new_q[1:]/np.sqrt(1 - new_q[0]**2)
        return ax, ang

    # Get rotation matrix
    def get_rotation_matrix(self):
        nq = self.total / np.dot(self.total, self.total)
        return np.array([[1-2*(nq[2]**2 + nq[3]**2), 2*(nq[1]*nq[2]-nq[3]*nq[0]), 2*(nq[1]*nq[3]+nq[2]*nq[0])],
                         [2*(nq[1]*nq[2]+nq[3]*nq[0]), 1-2*(nq[1]**2 + nq[3]**2), 2*(nq[2]*nq[3]-nq[1]*nq[0])],
                         [2*(nq[1]*nq[3]-nq[2]*nq[0]), 2*(nq[2]*nq[3]+nq[1]*nq[0]), 1-2*(nq[1]**2 + nq[2]**2)]])

    # Add two quaternions
    def __add__(self, other):
        return Quaternion(self.total + other.total)

    # Subtract two quaternions
    def __sub__(self, other):
        return Quaternion(self.total - other.total)

    # Multiply two quaternions
    def __mul__(self, other):
        return self.quat_prod_basic(other)

    # Print a quaternion
    def __repr__(self):
        return "q0=" + str(self.scalar) + ", q1=" + str(self.vector[0]) + ", q2=" + str(self.vector[1]) \
               + ", q3=" + str(self.vector[2])


# Dual-Quaternion Class
class DualQuaternion(object):
    def __init__(self, qr, qd):
        # qr and qd are two quaternions
        self.real = qr
        self.dual = qd

    # Get Conjugate of a Dual Quaternion
    def get_conjugate(self):
        real_conjugate = self.real.conjugate_quat()
        dual_conjugate = self.dual.conjugate_quat()
        return DualQuaternion(real_conjugate, dual_conjugate)

    # Print a dual quaternion
    def __repr__(self):
        real_str = "q0=" + str(self.real.scalar) + ", q1=" + str(self.real.vector[0]) + ", q2=" \
                   + str(self.real.vector[1]) + ", q3=" + str(self.real.vector[2])
        dual_str = "q0=" + str(self.dual.scalar) + ", q1=" + str(self.dual.vector[0]) + ", q2=" \
                   + str(self.dual.vector[1]) + ", q3=" + str(self.dual.vector[2])
        return "real: " + real_str + "\n" + "dual: " + dual_str

    # Add Two Dual-Quaternions
    def __add__(self, other):
        q_real = self.real + other.real
        q_dual = self.dual + other.dual
        return DualQuaternion(q_real, q_dual)

    # Subtract Two Dual-Quaternions
    def __sub__(self, other):
        q_real = self.real - other.real
        q_dual = self.dual - other.dual
        return DualQuaternion(q_real, q_dual)

    # Product of Two Dual-Quaternions
    def __mul__(self, other):
        q_real = self.real * other.real
        q_dual = (self.real * other.dual) + (self.dual * other.real)
        return DualQuaternion(q_real, q_dual)


# Class to deal Rigid-Displacements with Dual-Quaternions
# class Homo2DualQuat(object):
#     def __init__(self, p, rotm):
#         self.t = p
#         self.u, self.th = self.Rotm2AxisAng(rotm)
#         self.dual_quat = self.compute_dual_quat()
#         self.screw_d = self.compute_screw_d()
#         self.screw_m = self.compute_screw_m()
#
#     # Compute Dual-Quaternion
#     def compute_dual_quat(self):
#         real = self.AxisAng2Quat(self.u, self.th)
#         scalar_part = -np.sin(self.th/2)*np.dot(self.t, self.u)
#         vector_part = np.cos(self.th/2)*self.t + np.sin(self.th/2)*np.cross(self.t, self.u)
#         total_vec = np.hstack((scalar_part, vector_part))
#         dual = Quaternion(0.5*total_vec)
#         return DualQuaternion(real, dual)
#
#     # Compute d of screw parameter
#     def compute_screw_d(self):
#         temp = self.dual_quat.dual * self.dual_quat.real.conjugate_quat()
#         return 2 * np.dot(temp.vector, self.u)
#
#     # Compute m of screw parameter
#     def compute_screw_m(self):
#         temp1 = np.cross(self.t, self.u)
#         temp2 = (self.t - self.screw_d*self.u) / np.tan(self.th/2)
#         temp = temp1 + temp2
#         return 0.5 * temp
#
#     # Get Dual-Quaternion
#     def get_dual_quat(self):
#         return self.dual_quat
#
#     # Power of a dual-quaternion
#     def pow_dual_quat(self, tau):
#         real_vec = np.hstack((np.cos(tau * self.th/2), np.sin(tau * self.th/2) * self.u))
#         dual_vec = np.hstack((-tau * self.screw_d * np.sin(tau * self.th/2)/2, np.sin(tau * self.th/2) * self.screw_m + self.screw_d * np.cos(tau * self.th / 2) * self.u / 2))
#         return DualQuaternion(Quaternion(real_vec), Quaternion(dual_vec))
#
#     @staticmethod
#     def Rotm2AxisAng(rotm):
#         if (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 < -1 or (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 > 1:
#             raise Exception("Singular Rotation Configuration")
#         th = np.arccos((rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2)
#         temp = np.sqrt((rotm[2, 1] - rotm[1, 2]) ** 2 + (rotm[0, 2] - rotm[2, 0]) ** 2 + (rotm[1, 0] - rotm[0, 1]) ** 2)
#         wx = (rotm[2, 1] - rotm[1, 2]) / temp
#         wy = (rotm[0, 2] - rotm[2, 0]) / temp
#         wz = (rotm[1, 0] - rotm[0, 1]) / temp
#         u = np.array([wx, wy, wz])
#         return u, th
#
#     @staticmethod
#     def AxisAng2Quat(ax, ang):
#         qvec = np.array([np.cos(ang/2), np.sin(ang/2)*ax[0], np.sin(ang/2)*ax[1], np.sin(ang/2)*ax[2]])
#         return Quaternion(qvec)
#
#     @staticmethod
#     def Quat2AxisAng(qin):
#         if qin[0] == 1 or qin[0] == -1:
#             raise Exception("Singular rotation configuration")
#         th = 2 * np.arccos(qin[0])
#         u = qin[1:]/np.sqrt(1 - qin[0]**2)
#         return u, th


# Class to encapsulate screw parameters
class ScrewParam(object):
    def __init__(self, u, th, d, m):
        self.u = u
        self.th = th
        self.d = d
        self.m = m


# Class to interpolate between Rigid-Displacements in Dual-Quaternion form
class IntpRigidDispDualQuat(object):
    def __init__(self, p_i, q_i, p_f, q_f):
        self.initial_config = self.convert_dual_quat(p_i, q_i)
        self.final_config = self.convert_dual_quat(p_f, q_f)
        self.relative_config = self.initial_config.get_conjugate() * self.final_config

    # Given a Dual-Quaternion return its Screw parameters(u, th, d, m)
    def compute_screw_param(self, q):
        u, th = q.real.get_axis_angle()
        # u, th = self.Quat2AxisAng(q.real)
        d = self.compute_screw_d(q, u)
        temp = q.dual * q.real.conjugate_quat()
        t = 2 * temp.vector
        m = self.compute_screw_m(u, th, d, t)
        return ScrewParam(u, th, d, m)

    def interpolate(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        sp = self.compute_screw_param(self.relative_config)
        # p_initial = 2 * (self.initial_config.dual * self.initial_config.real.conjugate_quat()).vector
        # print("\nInitial Position")
        # print(p_initial)
        tau_list = np.linspace(0, 1, num=100)
        for idx, tau in enumerate(tau_list):
            dual_tau = self.initial_config * self.pow_dual_quat(sp, tau)
            if idx % 3 == 0 or idx == 0:
                print("Iteration %d" % idx)
                print("Intermediate Config (Dual-Quaternion):")
                print(dual_tau)

                position = 2 * (dual_tau.dual * dual_tau.real.conjugate_quat()).vector
                rotation_matrix = dual_tau.real.get_rotation_matrix()
                print("Intermediate Position (Vector form):")
                print(position)

                print("Intermediate Rotation (Matrix form):")
                print(rotation_matrix)

                print("\n")

                # Plot
                ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 0], rotation_matrix[1, 0],
                          rotation_matrix[2, 0], length=0.01, normalize=False, color='r')
                ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 1], rotation_matrix[1, 1],
                          rotation_matrix[2, 1], length=0.01, normalize=False, color='g')
                ax.quiver(position[0], position[1], position[2], rotation_matrix[0, 2], rotation_matrix[1, 2],
                          rotation_matrix[2, 2], length=0.01, normalize=False, color='b')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("Screw-Interpolation")
        # ax.set_aspect('equal')
        ax.view_init(10, 130)
        plt.show()

    # Convert Position Vector and Orientation Quaternion into Dual-Quaternion
    @staticmethod
    def convert_dual_quat(p, q):
        real = Quaternion(q)
        # u, th = self.Quat2AxisAng(real)
        u, th = real.get_axis_angle()
        scalar_part = -np.sin(th / 2) * np.dot(p, u)
        vector_part = np.cos(th / 2) * p + np.sin(th / 2) * np.cross(p, u)
        total_vec = np.hstack((scalar_part, vector_part))
        dual = Quaternion(0.5 * total_vec)
        return DualQuaternion(real, dual)

    # Power of a dual-quaternion
    @staticmethod
    def pow_dual_quat(screw_param, tau):
        real_vec = np.hstack((np.cos(tau * screw_param.th / 2), np.sin(tau * screw_param.th / 2) * screw_param.u))
        dual_vec = np.hstack((-tau * screw_param.d * np.sin(tau * screw_param.th / 2) / 2,
                              np.sin(tau * screw_param.th / 2) * screw_param.m + tau * screw_param.d * np.cos(
                                  tau * screw_param.th / 2) * screw_param.u / 2))
        return DualQuaternion(Quaternion(real_vec), Quaternion(dual_vec))

    # Compute d of screw parameter
    @staticmethod
    def compute_screw_d(dual_quat, u):
        temp = dual_quat.dual * dual_quat.real.conjugate_quat()
        return 2 * np.dot(temp.vector, u)

    # Compute m of screw parameter
    @staticmethod
    def compute_screw_m(u, th, d, t):
        temp1 = np.cross(t, u)
        temp2 = (t - d * u) / np.tan(th / 2)
        temp = temp1 + temp2
        return 0.5 * temp


# Rotation matrix to quaternion
def rotm2quat(rotm):
    if (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 < -1 or (rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2 > 1:
        raise Exception("Singular configuration to convert rotation matrix to quaternion")
    angle = np.arccos((rotm[0, 0] + rotm[1, 1] + rotm[2, 2] - 1) / 2)
    temp = np.sqrt((rotm[2, 1] - rotm[1, 2])**2 + (rotm[0, 2] - rotm[2, 0])**2 + (rotm[1, 0] - rotm[0, 1])**2)
    wx = (rotm[2, 1] - rotm[1, 2]) / temp
    wy = (rotm[0, 2] - rotm[2, 0]) / temp
    wz = (rotm[1, 0] - rotm[0, 1]) / temp
    quat = np.array([np.cos(angle / 2), wx * np.sin(angle / 2), wy * np.sin(angle / 2), wz * np.sin(angle / 2)])
    return quat


# Test cases
p_vec1 = np.array([0.1, 0.5, 0.23])
# p_vec2 = np.array([0.2, 0.6, 0.33])
p_vec2 = np.array([0.3, 0.7, 0.4])
ang1, ang2, ang3 = -np.pi/6, np.pi/3, -np.pi/4

rot_matrix1 = np.array([[np.cos(ang1), -np.sin(ang1), 0], [np.sin(ang1), np.cos(ang1), 0], [0, 0, 1]])
rot_matrix2 = np.array([[np.cos(ang2), 0, np.sin(ang2)], [0, 1, 0], [-np.sin(ang2), 0, np.cos(ang2)]])
rot_matrix3 = np.array([[1, 0, 0], [0, np.cos(ang3), -np.sin(ang3)], [0, np.sin(ang3), np.cos(ang3)]])
rot_matrix = np.dot(np.dot(rot_matrix1, rot_matrix2), rot_matrix3)

quat_vec1 = rotm2quat(rot_matrix1)
quat_vec2 = rotm2quat(rot_matrix1)

print("Initial Position: ")
print(p_vec1)
print("Initial Rotation Quaternion: ")
print(quat_vec1)

print("Final Position: ")
print(p_vec2)
print("Final Rotation Quaternion: ")
print(quat_vec2)
print("\n")

interpolate_obj = IntpRigidDispDualQuat(p_vec1, quat_vec1, p_vec2, quat_vec2)
interpolate_obj.interpolate()

