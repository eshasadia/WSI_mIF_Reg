"""
Evaluation functions for registration quality assessment
"""

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def tre(landmarks_1, landmarks_2):
    """
    Calculate Target Registration Error (TRE)
    
    Args:
        landmarks_1: First set of landmarks (N, 2)
        landmarks_2: Second set of landmarks (N, 2)
        
    Returns:
        TRE values for each landmark pair
    """
    tre_values = np.sqrt(
        np.square(landmarks_1[:, 0] - landmarks_2[:, 0]) + 
        np.square(landmarks_1[:, 1] - landmarks_2[:, 1])
    )
    return tre_values


def rtre(landmarks_1, landmarks_2, x_size, y_size):
    """
    Calculate relative Target Registration Error (rTRE)
    
    Args:
        landmarks_1: First set of landmarks (N, 2)
        landmarks_2: Second set of landmarks (N, 2)
        x_size: Image width
        y_size: Image height
        
    Returns:
        Relative TRE values
    """
    tre_values = tre(landmarks_1, landmarks_2)
    diagonal = np.sqrt(x_size * x_size + y_size * y_size)
    return tre_values / diagonal


def load_landmark_points(fixed_path, moving_path, scale_factor=1.0):
    """
    Load landmark points from CSV files
    
    Args:
        fixed_path: Path to fixed landmarks CSV
        moving_path: Path to moving landmarks CSV
        scale_factor: Scaling factor for coordinates
        
    Returns:
        tuple: (fixed_points, moving_points)
    """
    fixed_points = pd.read_csv(fixed_path, header=None, sep=',', skiprows=1).iloc[:, 1:].values
    moving_points = pd.read_csv(moving_path, header=None, sep=',', skiprows=1).iloc[:, 1:].values
    
    if scale_factor != 1.0:
        fixed_points *= scale_factor
        moving_points *= scale_factor
    
    return fixed_points, moving_points


def load_evaluation_landmarks(fixed_path, moving_path, scale_factor=1000):
    """
    Load evaluation landmark points (alternative format)
    
    Args:
        fixed_path: Path to fixed landmarks CSV
        moving_path: Path to moving landmarks CSV
        scale_factor: Scaling factor for coordinates
        
    Returns:
        tuple: (fixed_points, moving_points)
    """
    fixed_points = pd.read_csv(fixed_path, header=None).to_numpy()[:, :2] * scale_factor
    moving_points = pd.read_csv(moving_path, header=None).to_numpy()[:, :2] * scale_factor
    
    return fixed_points, moving_points


def transform_points_homogeneous(points, transform_matrix):
    """
    Transform points using homogeneous coordinates
    
    Args:
        points: Input points (N, 2)
        transform_matrix: 3x3 transformation matrix
        
    Returns:
        Transformed points (N, 2)
    """
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = (transform_matrix @ points_homogeneous.T).T
    return transformed_points[:, :2]


def evaluate_registration_tre(fixed_points, moving_points, transform_matrix, 
                             target_shape, scale_factor=None):
    """
    Evaluate registration using TRE metrics
    
    Args:
        fixed_points: Fixed landmark points
        moving_points: Moving landmark points
        transform_matrix: Transformation matrix to apply
        target_shape: Shape of target image for rTRE calculation
        scale_factor: Optional scaling factor for transform
        
    Returns:
        dict: Dictionary with TRE metrics
    """
    # Scale transformation if needed
    if scale_factor is not None:
        pre_transform_s = transform_matrix.copy()
        pre_transform_s[0:2, 2] = pre_transform_s[0:2, 2] * scale_factor
        transform_matrix = pre_transform_s
    
    # Transform moving points
    transformed_moving_points = transform_points_homogeneous(moving_points, transform_matrix)
    
    # Calculate TRE metrics
    tre_init = np.mean(np.linalg.norm(fixed_points - moving_points, axis=1))
    tre_final = np.mean(np.linalg.norm(fixed_points - transformed_moving_points, axis=1))
    rtre_values = rtre(fixed_points, transformed_moving_points, target_shape[1], target_shape[0])
    
    return {
        'tre_initial': tre_init,
        'tre_final': tre_final,
        'rtre_mean': np.mean(rtre_values),
        'rtre_std': np.std(rtre_values),
        'transformed_points': transformed_moving_points
    }


def apply_displacement_field_to_points(points, displacement_field, pixel_scale=1.0):
    """
    Apply displacement field to a set of points
    
    Args:
        points: Input points (N, 2)
        displacement_field: Displacement field (H, W, 2)
        pixel_scale: Scale factor to convert points to pixel coordinates
        
    Returns:
        tuple: (transformed_points, valid_mask)
    """
    # Convert to pixel coordinates
    points_px = points / pixel_scale
    
    # Prepare interpolation coordinates
    x_coords = points_px[:, 0]
    y_coords = points_px[:, 1]
    
    # Check image shape
    H, W, _ = displacement_field.shape
    
    # Ensure coordinates are within image bounds
    valid_mask = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    
    # Filter valid points
    x_coords_valid = x_coords[valid_mask]
    y_coords_valid = y_coords[valid_mask]
    
    # Interpolate dx and dy from displacement field
    dx_interp = map_coordinates(displacement_field[:, :, 0], 
                               [y_coords_valid, x_coords_valid], order=1)
    dy_interp = map_coordinates(displacement_field[:, :, 1], 
                               [y_coords_valid, x_coords_valid], order=1)
    
    # Apply displacement
    moved_points = np.vstack([
        x_coords_valid + dx_interp,
        y_coords_valid + dy_interp
    ]).T
    
    # Convert back to original scale
    moved_points_scaled = moved_points * pixel_scale
    
    return moved_points_scaled, valid_mask


def evaluate_nonrigid_registration(fixed_points, moving_points, rigid_transform, 
                                 displacement_field, pixel_scale=16):
    """
    Evaluate non-rigid registration with displacement field
    
    Args:
        fixed_points: Fixed landmark points
        moving_points: Moving landmark points  
        rigid_transform: Rigid transformation matrix
        displacement_field: Non-rigid displacement field
        pixel_scale: Scale factor for pixel conversion
        
    Returns:
        dict: Evaluation metrics
    """
    # Apply rigid transformation first
    transformed_moving_points = transform_points_homogeneous(moving_points, 
                                                            np.linalg.inv(rigid_transform))
    
    # Apply displacement field
    moved_points, valid_mask = apply_displacement_field_to_points(
        transformed_moving_points, displacement_field, pixel_scale
    )
    
    # Calculate TRE for valid points only
    fixed_points_valid = fixed_points[valid_mask]
    
    tre_init = np.mean(np.linalg.norm(fixed_points - moving_points, axis=1))
    tre_rigid = np.mean(np.linalg.norm(fixed_points_valid - transformed_moving_points[valid_mask], axis=1))
    tre_nonrigid = np.mean(np.linalg.norm(fixed_points_valid - moved_points, axis=1))
    
    return {
        'tre_initial': tre_init,
        'tre_rigid': tre_rigid, 
        'tre_nonrigid': tre_nonrigid,
        'valid_points': np.sum(valid_mask),
        'total_points': len(valid_mask)
    }


import numpy as np

from skimage import color

def ngf_metric(fixed_image, moving_image, epsilon=0.01):
    """
    Calculate Normalized Gradient Field metric.
    Works well for multi-stain registration as it focuses on edge alignment.
    """
    # Compute gradients
    fixed_image=color.rgb2gray(fixed_image)
    moving_image=color.rgb2gray(moving_image)
    fx, fy = np.gradient(fixed_image)
    mx, my = np.gradient(moving_image)
    
    # Normalize gradients
    fixed_mag = np.sqrt(fx**2 + fy**2) + epsilon
    moving_mag = np.sqrt(mx**2 + my**2) + epsilon
    
    fx_norm = fx / fixed_mag
    fy_norm = fy / fixed_mag
    mx_norm = mx / moving_mag
    my_norm = my / moving_mag
    
    # Calculate dot product of normalized gradients
    dot_product = fx_norm * mx_norm + fy_norm * my_norm
    
    # NGF measure (higher is better)
    ngf = np.mean(dot_product**2)
    print("ngf", ngf)
    return ngf


