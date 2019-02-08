from sympy import *
from mpmath import radians


class InverseKinematicSolver(object):

    def __init__(self):

        # Create Modified DH parameters
        self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7 = symbols('q1:8')  # link offset
        self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 = symbols('a0:7')  # link length
        self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.alpha5, self.alpha6 = symbols('alpha0:7')  # twist angle
        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7 = symbols('q1:8')  # joint angle symbols

        # Create row pitch yaw parameters
        self.r, self.p, self.y = symbols('r p y')

        # Define Modified DH Transformation matrix
        self.DH_table = {self.alpha0:     0,  self.a0:      0, self.d1:    0.75, self.q1:        self.q1,
                         self.alpha1: -pi/2., self.a1:   0.35, self.d2:       0, self.q2: self.q2 - pi/2,
                         self.alpha2:     0,  self.a2:   1.25, self.d3:       0, self.q3:        self.q3,
                         self.alpha3: -pi/2., self.a3: -0.054, self.d4:     1.5, self.q4:        self.q4,
                         self.alpha4: pi/2.,  self.a4:      0, self.d5:       0, self.q5:        self.q5,
                         self.alpha5: -pi/2., self.a5:      0, self.d6:       0, self.q6:        self.q6,
                         self.alpha6:     0,  self.a6:      0, self.d7:   0.303, self.q7:         0}

        # Perform individual transformation matrices
        self.T0_1 = self.tf_matrix(self.alpha0, self.a0, self.d1, self.q1).subs(self.DH_table)
        self.T1_2 = self.tf_matrix(self.alpha1, self.a1, self.d2, self.q2).subs(self.DH_table)
        self.T2_3 = self.tf_matrix(self.alpha2, self.a2, self.d3, self.q3).subs(self.DH_table)
        self.T3_4 = self.tf_matrix(self.alpha3, self.a3, self.d4, self.q4).subs(self.DH_table)
        self.T4_5 = self.tf_matrix(self.alpha4, self.a4, self.d5, self.q5).subs(self.DH_table)
        self.T5_6 = self.tf_matrix(self.alpha5, self.a5, self.d6, self.q6).subs(self.DH_table)
        self.T6_EE = self.tf_matrix(self.alpha6, self.a6, self.d7, self.q7).subs(self.DH_table)

        # Perform overall transformation of end-effector with respect of inertia reference frame
        self.T0_EE = self.T0_1 * self.T1_2 * self.T2_3 * self.T3_4 * self.T4_5 * self.T5_6 * self.T6_EE

        # Correct reference frame
        self.R_corr = Matrix([[0, 0, 1.0, 0], [0, -1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0, 1.0]])
        self.T0_EE = self.T0_EE * self.R_corr

        # Perform rotational
        self.ROT_EE = self.rot_z(self.y) * self.rot_y(self.p) * self.rot_x(self.r)
        self.ROT_ERROR = self.rot_z(self.y).subs(self.y, radians(180)) * self.rot_y(self.p).subs(self.p, radians(-90))
        self.ROT_EE = self.ROT_EE * self.ROT_ERROR
        self.WC = None
        self.EE = None

    def theta_angles(self, px, py, pz, roll, pitch, yaw):
        self.ROT_EE = self.ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})

        self.EE = Matrix([[px],
                          [py],
                          [pz]])

        # Wrist Center position
        self.WC = self.EE - self.DH_table.get(self.d7) * self.ROT_EE[:, 2]
        x = sqrt(self.WC[0]**2 + self.WC[1]**2) - self.DH_table.get(self.a1)
        z = self.WC[2] - self.DH_table.get(self.d1)

        side_a = sqrt(self.DH_table.get(self.d4)**2 + self.DH_table.get(self.a3)**2)
        side_b = sqrt(x**2 + z**2)
        side_c = self.DH_table.get(self.a2)

        # Calculate three inner angles
        alpha = acos((side_b**2 + side_c**2 - side_a**2)/(2 * side_b * side_c))
        beta = acos((side_a**2 + side_c**2 - side_b**2)/(2 * side_a * side_c))
        # gamma = acos((side_a**2 + side_b**2 - side_c**2)/(2 * side_a * side_b))

        # Calculate theta1, 2, 3
        theta1 = atan2(self.WC[1], self.WC[0])
        theta2 = pi/2 - alpha - atan2(z, x)
        theta3 = pi/2 - beta - atan2(self.DH_table.get(self.a3), self.DH_table.get(self.d4))

        # Calculate partial rotation matrix R3_6
        R0_3 = self.T0_1[0:3, 0:3] * self.T1_2[0:3, 0:3] * self.T2_3[0:3, 0:3]
        R0_3 = R0_3.evalf(subs={self.q1: theta1, self.q2: theta2, self.q3: theta3})
        R3_6 = R0_3.inv("LU") * self.ROT_EE

        # Obtain the Euler angles for the orientation of the wrist center from R3_6
        theta5 = atan2(sqrt(R3_6[0, 2] ** 2 + R3_6[2, 2] ** 2), R3_6[1, 2])

        if sin(theta5) < 0:
            theta4 = atan2(-R3_6[2, 2], R3_6[0, 2])
            theta6 = atan2(R3_6[1, 1], -R3_6[1, 0])
        else:
            theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
            theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])

        return theta1, theta2, theta3, theta4, theta5, theta6

    @staticmethod
    def tf_matrix(alpha, a, d, q):

        return Matrix([[cos(q),                      -sin(q),           0,             a],
                       [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                       [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                       [0,                                  0,          0,              1]])

    @staticmethod
    def rot_x(r):

        return Matrix([[1,      0,       0],
                       [0, cos(r), -sin(r)],
                       [0, sin(r), cos(r)]])

    @staticmethod
    def rot_y(p):

        return Matrix([[cos(p),      0,       sin(p)],
                       [0,           1,            0],
                       [0,      sin(p),      cos(p)]])

    @staticmethod
    def rot_z(y):

        return Matrix([[cos(y),      -sin(y),       0],
                       [sin(y),       cos(y),       0],
                       [0,                 0,       1]])
