import scipy
import numpy
import quaternion
import quaternion.quaternion_time_series

def IMU_double_integration(t, gyroscope, acceleration, t_out=None):
    """
    Compute position and orientation by integrating angular velocity and double integrating acceleration
    Expect the drift to be as large as hell
    :param t: time sequence, Nx1 array
    :param gyroscope: angular velocity, either Nx3 array (eular angle) or Nx4 array (quaternion)
    :param acceleration: acceleration data, Nx3 array
    :return: position: Nx3 array, orientation (quaternion): Nx4 array
    """
    if gyroscope.shape[1] == 4:
        eular = [quaternion.as_euler_angles(quaternion.quaternion(*q)) for q in gyroscope]
    else:
        eular = gyroscope

    if t_out == None:
        t_out = t
