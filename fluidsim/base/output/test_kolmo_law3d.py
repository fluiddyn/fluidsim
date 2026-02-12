"""
This file contains unit tests for kolmo_laws3d.py file
For coordinate changes, the following vectors are tested:
(1, 1, 1)
expected cylindrical: (sqrt(2), pi/4 rad = 45°, 1)
(-1, -1, -1)
expected cylindrical: (sqrt(2), -3pi/4 rad = -135°, -1)
(1, 0, 0)
expected cylindrical: (1, 0 rad = 0°, 0)
(0, 1, 0)
expected cylindrical: (1, pi/2 rad = 90°, 0)
(0, 0, 1)
expected cylindrical: (0, 0 rad = 0°, 1)

Every points that respect y = tan(pi/3)x has the same rtheta coordinate at pi/3  rad = 60° when x>0 and -4pi/3 rad = -120° when x<0
These points are on a plane orthogonal to the (x,y) plane and with an angle theta with the plane (x,z)
"""

import numpy as np
from math import pi, degrees, tan


from kolmo_law3d import OperKolmoLaw


def get_cylindrical_components(x, y, z):
    oper_kolmo_law = OperKolmoLaw(x, y, z)
    return oper_kolmo_law.compute_cylindrical_components()


def get_bool_test(r, theta, z, r_expect, theta_expect, z_expect):
    if (
        (round(r, ndigits=10) == round(r_expect, ndigits=10))
        and (round(theta, ndigits=10) == round(theta_expect[0], ndigits=10))
        and (round(z, ndigits=10) == round(z_expect, ndigits=10))
    ):
        return True
    else:
        return False


def print_and_calculate_quantities(
    x, y, z, r_expect, theta_expect, z_expect, print_test=True
):
    if print_test:
        print(
            f"Testing vector ({x}, {y}, {z}), expecting result: ({r_expect}, {theta_expect[0]} rad = {theta_expect[1]} rad = {theta_expect[2]}°, {z_expect})"
        )
    (r, theta, z) = get_cylindrical_components(x, y, z)
    if print_test:
        print(f"Result: {(r, theta, z) = }, {degrees(theta)=}°")
    else:
        print(
            f"r/x = {r/x}, theta = {theta} rad = {degrees(theta)}° and z/z = {z/z_expect}"
        )
    return r, theta, z


def unit_tests_cartesian_to_cylindrical():
    Test = True
    num_error = 0
    print(
        "Testing conversion from cartesian coordinates system to cylindrical in kolmo_laws3d.py"
    )
    xyz_list = [[1, 1, 1], [-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    expected_list = [
        [np.sqrt(2), [pi / 4, "pi/4", 45], 1],
        [np.sqrt(2), [-(3 * pi) / 4, "-3pi/4", -135], -1],
        [1, [0, "0", 0], 0],
        [1, [pi / 2, "pi/2", 90], 0],
        [0, [0, "0", 0], 1]
    ]

    def new_test(xyz, expected, num_error, Test, print_test=True):
        x, y, z = xyz[:]
        r_expect, theta_expect, z_expect = expected[:]
        r, theta, z = print_and_calculate_quantities(
            x=x, y=y, z=z, r_expect=r_expect, theta_expect=theta_expect, z_expect=z_expect, print_test=print_test
        )
        Test = get_bool_test(r, theta, z, r_expect, theta_expect, z_expect)
        if not Test:
            print("Error in test")
            num_error += 1
        return num_error

    for i in range(len(xyz_list)):
        xyz = xyz_list[i]
        expected = expected_list[i]
        num_error = new_test(xyz, expected, num_error, Test)
        
    print(
        f"Testing vectors such that y = tan(pi/3)x  expecting result: z/z = 1 and either r/x = {np.sqrt(1+tan(pi/3)**2)} and theta = pi/3 rad = {pi/3} rad = 60° if x>0 or r/x = -{np.sqrt(1+tan(pi/3)**2)} and theta = -2pi/3 rad = {-(2*pi)/3} rad = -120° if x<0"
    )
    x_rand = np.linspace(-(np.random.rand() * 100), np.random.rand() * 100, 10)
    z_rand = np.linspace(-(np.random.rand() * 100), np.random.rand() * 100, 4)
    for x in x_rand:
        for z_test in z_rand:
            xyz = [x, tan(pi / 3) * x, z_test]
            if x > 0:
                expected = [
                    np.sqrt(1 + tan(pi / 3) ** 2) * x,
                    [pi / 3, "pi/3", 60],
                    z_test,
                ]
            elif x < 0:
                expected = [
                    -np.sqrt(1 + tan(pi / 3) ** 2) * x,
                    [-(2 * pi) / 3, "-2pi/3", -120],
                    z_test,
                ]
            else:
                expected = [0, [0, "0", 0], z_test]
            num_error = new_test(xyz, expected, num_error, Test, print_test=False)
    print(f"{x_rand = }")
    print(f"{z_rand = }")
    if num_error == 0:
        print("ALL TESTS ARE GOOD !")
    else:
        raise ValueError(f"SOME TEST WENT WRONG ! There are {num_error} erros.")


def make_all_tests():
    unit_tests_cartesian_to_cylindrical()


if __name__ == "__main__":
    make_all_tests()
