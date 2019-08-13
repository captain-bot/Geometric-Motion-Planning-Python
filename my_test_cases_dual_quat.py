from dual_quat_library import *

# Test Dual-Quaternion Class
a = np.array([0.9239, 0, 0, 0.3827])
b = np.array([0.8660, 0, 0, 0.5000])
c = np.array([0.9659, 0, 0, 0.2588])

test_quat1 = Quaternion(a)
test_quat2 = Quaternion(b)
test_quat3 = Quaternion(c)

test_dual_quat1 = DualQuaternion(test_quat1, test_quat2)
print("\nDual-Quaternion")
print(test_dual_quat1)

test_dual_quat2 = DualQuaternion(test_quat2, test_quat3)
print("\nDual-Quaternion")
print(test_dual_quat2)

test_dual_quat3 = test_dual_quat1 * test_dual_quat2
print("\nProduct of two Dual-Quaternions")
print(test_dual_quat3)

test_dual_quat4 = test_dual_quat1 + test_dual_quat2
print("\nAddition of two Dual-Quaternions")
print(test_dual_quat4)
