from dual_quat_library import *

# Test Quaternion Class
a = np.array([0.9239, 0, 0, 0.3827])
b = np.array([0.8660, 0, 0, 0.5000])
c = np.array([0.0000, 0, 0, 1.0000])

test_quat1 = Quaternion(a)
test_quat2 = Quaternion(b)
test_quat3 = Quaternion(c)

test_conjgate_quat = test_quat1.conjugate_quat()

print(test_quat1)
print(test_quat2)
print(test_quat3)
print(test_conjgate_quat)

axis, angle = test_quat1.get_axis_angle()
print("\naxis: ", axis)
print("angle: ", angle)

axis, angle = test_quat3.get_axis_angle()
print("\naxis: ", axis)
print("angle: ", angle)

rot_mat1 = test_quat1.get_rotation_matrix()
print(rot_mat1)

rot_mat2 = test_quat2.get_rotation_matrix()
print(rot_mat2)

print("\nAddition of quaternions")
print(test_quat1 + test_quat2)

print("\nSubtraction of quaternions")
print(test_quat1 - test_quat2)

print("\nProduct of quaternions (Method 1)")
print(test_quat1.quat_prod(test_quat2))

print("\nProduct of quaternions (Method 2)")
print(test_quat1.quat_prod_basic(test_quat2))

print("\nProduct of quaternions (Method 3)")
print(test_quat1 * test_quat2)
