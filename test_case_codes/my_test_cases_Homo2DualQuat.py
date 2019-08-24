import numpy as np
from dual_quat_library import Homo2DualQuat

# Test cases
p_vec = np.array([0.1, 0.5, 0.23])
ang1, ang2, ang3 = -np.pi/6, np.pi/3, -np.pi/4

rot_matrix1 = np.array([[np.cos(ang1), -np.sin(ang1), 0], [np.sin(ang1), np.cos(ang1), 0], [0, 0, 1]])
rot_matrix2 = np.array([[np.cos(ang2), 0, np.sin(ang2)], [0, 1, 0], [-np.sin(ang2), 0, np.cos(ang2)]])
rot_matrix3 = np.array([[1, 0, 0], [0, np.cos(ang3), -np.sin(ang3)], [0, np.sin(ang3), np.cos(ang3)]])
rot_matrix = np.dot(np.dot(rot_matrix1, rot_matrix2), rot_matrix3)
rigid_displacement = Homo2DualQuat(p_vec, rot_matrix)
dq = rigid_displacement.get_dual_quat()
print(dq)

# Get the conjugate of dq
cdq = dq.get_conjugate()
print("\n")
print(cdq)

# Check Unit-Dual-Quaternion property
# (i.e., if qd represents rigid displacement then dq * conjugate(dq) = unit dual quaternion)
c_prod = dq * cdq
print("\n")
print(c_prod)

# Retrieve p_vec from dual quaternion representation
p = dq.real
q = dq.dual
linear_displacement = 2 * (q * p.conjugate_quat()).vector
print("\nObtained p_vec: ")
print(linear_displacement)

# Compute d of screw parameter
d = rigid_displacement.compute_screw_d()
print("\nScrew parameter: d")
print(d)

# Compute m of screw parameter
m = rigid_displacement.compute_screw_m()
print("\nScrew parameter: m")
print(m)

# Compute power of a dual quaternion
tau = 0.1
rigid_displacement.pow_dual_quat(tau)