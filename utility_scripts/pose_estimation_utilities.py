import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_rotation_matrix_from_ortho6d(ortho6d):
    """
    Code adapted from Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    Input: batch_size x 6
    Output: batch_size x 3 x 3
    """
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    # Normalize the first vector (x)
    x = F.normalize(x_raw, dim=-1, eps = 1e-6)
    
    # Make y orthogonal to x
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps = 1e-6)
    
    # Recalculate y to be orthogonal to x and z
    y = torch.cross(z, x, dim = -1)

    # Stack them to form the matrix
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    
    matrix = torch.cat((x, y, z), 2) # Batch x 3 x 3
    return matrix


def euler_angles_to_matrix(euler_angles, to_radians=True):
    """
    Converts Euler angles to a 3x3 Rotation Matrix.
    
    Args:
        euler_angles (Tensor): (B, 3) tensor containing (Roll, Pitch, Yaw).
        to_radians (bool): If True, converts input from degrees to radians first.
        
    Returns:
        rotation_matrix (Tensor): (B, 3, 3)
    """
    # 1. Convert to Radians if necessary
    if to_radians:
        euler_angles = torch.deg2rad(euler_angles)
        
    roll  = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw   = euler_angles[:, 2]

    # 2. Precompute sines and cosines
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    # 3. Define the 3 rotation matrices
    # Rx (Pitch)
    # | 1  0   0 |
    # | 0 cp -sp |
    # | 0 sp  cp |
    
    # Ry (Yaw)
    # | cy 0 sy |
    # |  0 1  0 |
    # |-sy 0 cy |
    
    # Rz (Roll)
    # | cr -sr 0 |
    # | sr  cr 0 |
    # |  0   0 1 |

    # 4. Compute Combined Matrix R = Rz * Ry * Rx
    # We compute elements manually for efficiency
    
    m00 = cr * cy
    m01 = cr * sy * sp - sr * cp
    m02 = cr * sy * cp + sr * sp
    
    m10 = sr * cy
    m11 = sr * sy * sp + cr * cp
    m12 = sr * sy * cp - cr * sp
    
    m20 = -sy
    m21 = cy * sp
    m22 = cy * cp

    # Stack into (B, 3, 3)
    row1 = torch.stack([m00, m01, m02], dim=1)
    row2 = torch.stack([m10, m11, m12], dim=1)
    row3 = torch.stack([m20, m21, m22], dim=1)
    
    matrix = torch.stack([row1, row2, row3], dim=1)
    
    return matrix


def matrix_to_euler_angles(matrix, to_degrees=True):
    """
    Converts a 3x3 Rotation Matrix to Euler angles (Roll, Pitch, Yaw).
     Assumes the R = Rz * Ry * Rx convention used above.
    
    Args:
        matrix (Tensor): (B, 3, 3) rotation matrix.
        degrees (bool): If True, converts output to degrees.
        
    Returns:
        euler_angles (Tensor): (B, 3) tensor containing (Roll, Pitch, Yaw).
    """
    # Clamp to ensure numerical stability for arcsin (values must be between -1 and 1)
    m20 = torch.clamp(matrix[:, 2, 0], -1.0, 1.0)
    m21 = matrix[:, 2, 1]
    m22 = matrix[:, 2, 2]
    m00 = matrix[:, 0, 0]
    m10 = matrix[:, 1, 0]

    # Sy_sqrt = cos(yaw)
    sy_sqrt = torch.sqrt(m21 * m21 + m22 * m22)

    # We generally check if cos(yaw) is close to zero (Gimbal Lock case)
    # However, for batch processing with head pose, we typically assume standard case:
    
    # Yaw (rotation around Y)
    yaw = torch.atan2(-m20, sy_sqrt)
    
    # Pitch (rotation around X)
    pitch = torch.atan2(m21, m22)
    
    # Roll (rotation around Z)
    roll = torch.atan2(m10, m00)

    # Stack results
    euler_angles = torch.stack([roll, pitch, yaw], dim=1)
    
    if to_degrees:
        euler_angles = torch.rad2deg(euler_angles)
        
    return euler_angles


import cv2
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50, degrees = True):
    """
    img: numpy array (H, W, 3) or (H, W) - Input image
    yaw, pitch, roll: angles in degrees
    tdx, tdy: Center of the face (x, y). If None, uses image center.
    size: Length of the axis lines
    """
    # Convert degrees to radians
    if degrees:
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180) 
        roll = roll * np.pi / 180

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in Red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in Green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of screen) drawn in Blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # Draw the lines
    # OpenCV uses BGR, so (0,0,255) is Red, (0,255,0) is Green
    img = img.copy()
    
    # Pitch (X-axis) - Red
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(255, 0, 0),2)
    # Yaw (Y-axis) - Green
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    # Roll (Z-axis) - Blue
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0, 0, 255),2)

    return img