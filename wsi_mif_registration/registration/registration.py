"""
Registration algorithms for WSI alignment
"""

import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from pycpd import DeformableRegistration
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.optimize import minimize
import wsi_mif_registration.registration.rigid as rigid


def perform_rigid_registration(source_prep, target_prep, source_mask, target_mask):
    """
    Perform rigid registration between source and target images
    
    Args:
        source_prep: Preprocessed source image
        target_prep: Preprocessed target image
        source_mask: Source tissue mask
        target_mask: Target tissue mask
        
    Returns:
        tuple: (transformed_image, transformation_matrix)
    """
    moving_img_transformed, final_transform,_ = rigid.rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )
    
    # Apply transformation
    height, width = source_prep.shape[:2]
    warped = cv2.warpAffine(source_prep, final_transform[0:-1], (width, height))
    
    return  warped , final_transform


def perform_icp_registration(moving_points, fixed_points, threshold=50000.0):
    """
    Perform ICP registration on point sets
    
    Args:
        moving_points: Moving point set (N, 2)
        fixed_points: Fixed point set (M, 2)
        threshold: Maximum correspondence distance
        
    Returns:
        tuple: (transformation_matrix, transformed_points)
    """
    # Convert to Open3D point clouds (add z=0 for 2D)
    pcd_moving = o3d.geometry.PointCloud()
    pcd_moving.points = o3d.utility.Vector3dVector(
        np.hstack([moving_points, np.zeros((len(moving_points), 1))])
    )
    
    pcd_fixed = o3d.geometry.PointCloud()
    pcd_fixed.points = o3d.utility.Vector3dVector(
        np.hstack([fixed_points, np.zeros((len(fixed_points), 1))])
    )
    
    # Run ICP registration
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_moving, pcd_fixed, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    print("ICP Transformation Matrix:")
    print(reg_p2p.transformation)
    
    # Apply transformation to moving points
    moving_points_hom = np.hstack([
        moving_points,
        np.zeros((len(moving_points), 1)),  # z=0
        np.ones((len(moving_points), 1))    # homogeneous coordinate
    ])
    
    transformed_points = (reg_p2p.transformation @ moving_points_hom.T).T[:, :2]
    
    return reg_p2p.transformation, transformed_points

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

class ShapeAwarePointSetRegistration:
    def __init__(self, fixed_points, moving_points,
                 shape_attribute='area', shape_weight=0.3,
                 max_iterations=50, tolerance=1e-6):
        self.fixed = fixed_points.copy()
        self.moving = moving_points.copy()
        self.shape_attribute = shape_attribute
        self.shape_weight = shape_weight
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._normalize_shape_attributes()

        self.rotation = 0.0
        self.scale = 1.0
        self.translation = np.zeros(2)

    def _normalize_shape_attributes(self):
        if self.shape_attribute in self.fixed.columns and self.shape_attribute in self.moving.columns:
            all_vals = np.concatenate([
                self.fixed[self.shape_attribute].values,
                self.moving[self.shape_attribute].values
            ])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            range_val = max_val - min_val if max_val > min_val else 1.0
            self.fixed['norm_shape'] = (self.fixed[self.shape_attribute] - min_val) / range_val
            self.moving['norm_shape'] = (self.moving[self.shape_attribute] - min_val) / range_val
        else:
            self.shape_weight = 0
            self.fixed['norm_shape'] = 0.5
            self.moving['norm_shape'] = 0.5

    def _apply_transform(self, points, R, t, s):
        return s * (points @ R.T) + t

    def _estimate_rigid_transform(self, A, B):
        """
        Estimate similarity transform (rotation, scale, translation) from A → B.
        """
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = np.trace(BB.T @ AA @ R) / np.trace(AA.T @ AA)
        t = centroid_B - scale * R @ centroid_A

        return R, t, scale

    def register(self):
        fixed_xy = self.fixed[['global_x', 'global_y']].values
        fixed_shape = self.fixed['norm_shape'].values

        moving_xy = self.moving[['global_x', 'global_y']].values
        moving_shape = self.moving['norm_shape'].values

        R = np.eye(2)
        t = np.zeros(2)
        s = 1.0

        prev_error = np.inf

        for i in range(self.max_iterations):
            transformed = self._apply_transform(moving_xy, R, t, s)
            kdtree = KDTree(fixed_xy)
            dists, idx = kdtree.query(transformed)

            matched_fixed = fixed_xy[idx]
            shape_dists = np.abs(moving_shape - fixed_shape[idx])
            total_dists = (1 - self.shape_weight) * dists + self.shape_weight * shape_dists

            error = np.mean(total_dists)

            # Estimate new transformation from matched pairs
            R_new, t_new, s_new = self._estimate_rigid_transform(moving_xy, matched_fixed)

            # Update transformation
            R = R_new
            t = t_new
            s = s_new

            if np.abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        # Final transformation
        transformed = self._apply_transform(moving_xy, R, t, s)
        self.registered_points = self.moving.copy()
        self.registered_points['registered_x'] = transformed[:, 0]
        self.registered_points['registered_y'] = transformed[:, 1]

        # Store final params
        self.rotation = np.arctan2(R[1, 0], R[0, 0])
        self.translation = t
        self.scale = s
        self.final_error = prev_error

        return self.registered_points

    def get_transformation_matrix(self):
        theta = self.rotation
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        tx, ty = self.translation
        s = self.scale
        return np.array([
            [s * cos_t, -s * sin_t, tx],
            [s * sin_t,  s * cos_t, ty],
            [0,          0,         1]
        ])


def perform_shape_aware_registration(fixed_df, moving_df, shape_attribute='area',
                                     shape_weight=0.3, max_iterations=50, tolerance=1e-6):
    registrator = ShapeAwarePointSetRegistration(
        fixed_df, moving_df,
        shape_attribute=shape_attribute,
        shape_weight=shape_weight,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    registered_points = registrator.register()
    transform_matrix = registrator.get_transformation_matrix()
    coords = registered_points[['registered_x', 'registered_y']].values
    return registrator, transform_matrix, coords



# class ShapeAwarePointSetRegistration:
#     """
#     Implementation of shape-aware point set registration that takes into account
#     both spatial coordinates and shape attributes (like area) of each point.
#     Supports both rigid and non-rigid registration.
#     """
   
#     def __init__(self, fixed_points, moving_points, shape_attribute='area',
#                  shape_weight=0.3, max_iterations=100, tolerance=1e-6,
#                  use_nonrigid=True, regularization=1.0, smoothness=2.0):
#         """
#         Initialize the registration algorithm.
       
#         Parameters:
#         -----------
#         fixed_points : pandas.DataFrame
#             DataFrame containing the fixed point set with columns 'global_x', 'global_y' and shape_attribute
#         moving_points : pandas.DataFrame
#             DataFrame containing the moving point set with columns 'global_x', 'global_y' and shape_attribute
#         shape_attribute : str
#             Column name for the shape attribute to consider (e.g., 'area')
#         shape_weight : float
#             Weight for the shape attribute in the distance calculation (between 0 and 1)
#         max_iterations : int
#             Maximum number of iterations for the registration
#         tolerance : float
#             Convergence tolerance
#         use_nonrigid : bool
#             Whether to perform non-rigid registration after rigid alignment
#         regularization : float
#             Regularization strength for non-rigid registration. Higher values preserve shape better.
#         smoothness : float
#             Smoothness parameter for the non-rigid transformation. Higher values give smoother transformations.
#         """
#         # Store input data
#         self.fixed_points = fixed_points.copy()
#         self.moving_points = moving_points.copy()
#         self.shape_attribute = shape_attribute
#         self.shape_weight = shape_weight
#         self.max_iterations = max_iterations
#         self.tolerance = tolerance
       
#         # Non-rigid parameters
#         self.use_nonrigid = use_nonrigid
#         self.regularization = regularization
#         self.smoothness = smoothness
       
#         # Normalized shape attributes for both point sets
#         self._normalize_shape_attributes()
       
#         # Initialize transformation parameters
#         self.translation = np.zeros(2)
#         self.rotation = 0.0
#         self.scale = 1.0
       
#         # Non-rigid transformation function
#         self.nonrigid_transform_x = None
#         self.nonrigid_transform_y = None
       
#         # Results
#         self.registered_points = None
#         self.correspondence_indices = None
#         self.final_error = None
#         self.rigid_registered_points = None
       
#     def _normalize_shape_attributes(self):
#         """Normalize shape attributes to [0, 1] range for both point sets"""
#         if self.shape_attribute in self.fixed_points.columns and self.shape_attribute in self.moving_points.columns:
#             # Find global min and max for normalization
#             all_values = np.concatenate([
#                 self.fixed_points[self.shape_attribute].values,
#                 self.moving_points[self.shape_attribute].values
#             ])
#             min_val = np.min(all_values)
#             max_val = np.max(all_values)
#             range_val = max_val - min_val
           
#             if range_val > 0:
#                 self.fixed_points[f'normalized_{self.shape_attribute}'] = (
#                     (self.fixed_points[self.shape_attribute] - min_val) / range_val
#                 )
#                 self.moving_points[f'normalized_{self.shape_attribute}'] = (
#                     (self.moving_points[self.shape_attribute] - min_val) / range_val
#                 )
#             else:
#                 # If all values are the same, set normalized value to 0.5
#                 self.fixed_points[f'normalized_{self.shape_attribute}'] = 0.5
#                 self.moving_points[f'normalized_{self.shape_attribute}'] = 0.5
#         else:
#             # If shape attribute is not available, don't use shape information
#             print(f"Warning: Shape attribute '{self.shape_attribute}' not found in one or both point sets.")
#             self.shape_weight = 0
#             self.fixed_points[f'normalized_{self.shape_attribute}'] = 0.5
#             self.moving_points[f'normalized_{self.shape_attribute}'] = 0.5
   
#     def _apply_transformation(self, points, params=None):
#         """
#         Apply transformation to points.
       
#         Parameters:
#         -----------
#         points : numpy.ndarray
#             Points to transform, shape (N, 2)
#         params : numpy.ndarray, optional
#             Transformation parameters [tx, ty, rotation, scale]
           
#         Returns:
#         --------
#         numpy.ndarray
#             Transformed points, shape (N, 2)
#         """
#         if params is None:
#             tx, ty = self.translation
#             theta = self.rotation
#             s = self.scale
#         else:
#             tx, ty, theta, s = params
       
#         # Extract coordinates
#         x, y = points[:, 0], points[:, 1]
       
#         # Calculate transformed coordinates
#         cos_theta, sin_theta = np.cos(theta), np.sin(theta)
#         x_transformed = s * (x * cos_theta - y * sin_theta) + tx
#         y_transformed = s * (x * sin_theta + y * cos_theta) + ty
       
#         return np.column_stack((x_transformed, y_transformed))
   
#     def _find_correspondences(self, transformed_moving_points):
#         """
#         Find correspondences between transformed moving points and fixed points
#         using shape-aware distance.
       
#         Parameters:
#         -----------
#         transformed_moving_points : numpy.ndarray
#             Transformed moving points, shape (N, 2)
           
#         Returns:
#         --------
#         tuple
#             (correspondence_indices, mean_distance)
#         """
#         # Get shape attributes
#         fixed_shape = self.fixed_points[f'normalized_{self.shape_attribute}'].values
#         moving_shape = self.moving_points[f'normalized_{self.shape_attribute}'].values
       
#         # Get fixed points coordinates
#         fixed_coords = self.fixed_points[['global_x', 'global_y']].values
       
#         # Create KDTree for fixed points
#         kdtree = KDTree(fixed_coords)
       
#         # Find nearest neighbors based on spatial coordinates
#         distances, indices = kdtree.query(transformed_moving_points, k=1)
       
#         # Calculate shape differences
#         shape_diffs = np.abs(moving_shape - fixed_shape[indices])
       
        
#         combined_distances = (1 - self.shape_weight) * distances + self.shape_weight * shape_diffs
       
#         return indices, np.mean(combined_distances)
   
#     def _objective_function(self, params):
#         """
#         Objective function for optimization.
       
#         Parameters:
#         -----------
#         params : numpy.ndarray
#             Transformation parameters [tx, ty, rotation, scale]
           
#         Returns:
#         --------
#         float
#             Mean combined distance between corresponding points
#         """
#         # Get moving points coordinates
#         moving_coords = self.moving_points[['global_x', 'global_y']].values
       
#         # Apply transformation
#         transformed_moving_points = self._apply_transformation(moving_coords, params)
       
#         # Find correspondences and get mean distance
#         _, mean_distance = self._find_correspondences(transformed_moving_points)
       
#         return mean_distance
   
#     def get_transformation_matrix(self):
#         """
#         Returns the 3x3 homogeneous transformation matrix representing
#         the rigid transformation (scale, rotation, translation).

#         Returns:
#         --------
#         numpy.ndarray
#             3x3 transformation matrix
#         """
#         tx, ty = self.translation
#         theta = self.rotation
#         s = self.scale

#         # Rotation matrix with scaling
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)

#         transformation_matrix = np.array([
#             [s * cos_theta, -s * sin_theta, tx],
#             [s * sin_theta,  s * cos_theta, ty],
#             [0,              0,             1 ]
#         ])

#         return transformation_matrix
    
#     def register(self):
#         """
#         Perform shape-aware point set registration.
#         First applies rigid registration, then optionally non-rigid registration.
       
#         Returns:
#         --------
#         pandas.DataFrame
#             Moving points after registration
#         """
#         # Initial parameters [tx, ty, rotation, scale]
#         initial_params = np.array([0.0, 0.0, 0.0, 1.0])
       
#         # Optimize transformation parameters (rigid registration)
#         result = minimize(
#             self._objective_function,
#             initial_params,
#             method='Powell',
#             options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
#         )
       
#         # Store optimized parameters
#         self.translation = result.x[:2]
#         self.rotation = result.x[2]
#         self.scale = result.x[3]
#         self.final_error = result.fun
       
#         # Get moving points coordinates
#         moving_coords = self.moving_points[['global_x', 'global_y']].values
       
#         # Apply rigid transformation
#         rigid_transformed_coords = self._apply_transformation(moving_coords)
       
#         # Find correspondences after rigid registration
#         correspondence_indices, _ = self._find_correspondences(rigid_transformed_coords)
#         self.correspondence_indices = correspondence_indices
       
#         # Save the rigid registration result
#         self.rigid_registered_points = self.moving_points.copy()
#         self.rigid_registered_points['registered_x'] = rigid_transformed_coords[:, 0]
#         self.rigid_registered_points['registered_y'] = rigid_transformed_coords[:, 1]
  
#         # Use rigid registration result
#         self.registered_points = self.rigid_registered_points.copy()
       
#         print(f"Shape-aware registration completed with error: {self.final_error}")
#         print(f"Rigid transformation parameters:")
#         print(f"  Translation: ({self.translation[0]:.4f}, {self.translation[1]:.4f})")
#         print(f"  Rotation: {np.degrees(self.rotation):.4f} degrees")
#         print(f"  Scale: {self.scale:.4f}")
       
#         return self.registered_points


# def perform_shape_aware_registration(fixed_df, moving_df, shape_attribute='area', 
#                                    shape_weight=0.3, max_iterations=200, tolerance=1e-8):
#     """
#     Perform shape-aware point set registration
    
#     Args:
#         fixed_df: DataFrame with fixed points
#         moving_df: DataFrame with moving points  
#         shape_attribute: Column name for shape attribute (e.g., 'area')
#         shape_weight: Weight for shape attribute (0-1)
#         max_iterations: Maximum optimization iterations
#         tolerance: Convergence tolerance
        
#     Returns:
#         tuple: (registrator_object, transformation_matrix, transformed_points)
#     """
#     # Initialize registration algorithm
#     registrator = ShapeAwarePointSetRegistration(
#         fixed_df,
#         moving_df,
#         shape_attribute=shape_attribute,
#         shape_weight=shape_weight,
#         max_iterations=max_iterations,
#         tolerance=tolerance
#     )
   
#     # Perform registration
#     registered_points = registrator.register()
    
#     # Get transformation matrix
#     transform_matrix = registrator.get_transformation_matrix()
    
#     # Extract transformed coordinates
#     transformed_coords = registered_points[['registered_x', 'registered_y']].values
    
#     return registrator, transform_matrix, transformed_coords


def find_mutual_nearest_neighbors(fixed_points, moving_points):
    """
    Find mutual nearest neighbors between two point sets
    
    Args:
        fixed_points: Fixed point set (N, 2)
        moving_points: Moving point set (M, 2)
        
    Returns:
        tuple: (fixed_mnn, moving_mnn, mnn_pairs)
    """
    # Find nearest moving for each fixed
    nn_fixed_to_moving = NearestNeighbors(n_neighbors=1).fit(moving_points)
    dist1, idx1 = nn_fixed_to_moving.kneighbors(fixed_points)
    
    # Find nearest fixed for each moving
    nn_moving_to_fixed = NearestNeighbors(n_neighbors=1).fit(fixed_points)
    dist2, idx2 = nn_moving_to_fixed.kneighbors(moving_points)
    
    # Mutual nearest neighbors condition
    mnn_pairs = [(i, j[0]) for i, j in enumerate(idx1) if idx2[j[0]] == i]
    
    # Extract matched points
    fixed_mnn = np.array([fixed_points[i] for i, _ in mnn_pairs])
    moving_mnn = np.array([moving_points[j] for _, j in mnn_pairs])
    
    print(f"Matched MNN pairs: {len(mnn_pairs)}")
    
    return fixed_mnn, moving_mnn, mnn_pairs


def perform_cpd_registration(moving_points, fixed_points, beta=0.5, alpha=0.01, 
                           max_iterations=200, tolerance=1e-9):
    """
    Perform Coherent Point Drift (CPD) non-rigid registration
    
    Args:
        moving_points: Moving point set (N, 2)
        fixed_points: Fixed point set (M, 2)
        beta: Smoothness parameter
        alpha: Regularization parameter
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Transformed moving points
    """
    reg = DeformableRegistration(
        X=moving_points, 
        Y=fixed_points,
        beta=beta,
        alpha=alpha,
        max_iterations=max_iterations,
        tol=tolerance
    )
    
    reg.register()
    return reg.TY


def create_displacement_field(source_points, transformed_points, image_shape, 
                            method='linear', sigma=10.0, max_displacement=10.0):
    """
    Generate a dense displacement field from sparse CPD results
    
    Args:
        source_points: (N, 2) before non-rigid
        transformed_points: (N, 2) after non-rigid
        image_shape: (H, W, C) or (H, W)
        method: interpolation method ('linear', 'cubic', 'nearest')
        sigma: Gaussian smoothing factor
        max_displacement: clamp maximum pixel displacement
        
    Returns:
        displacement_field: (H, W, 2)
    """
    displacements = transformed_points - source_points
    
    if len(image_shape) == 3:
        height, width, _ = image_shape
    else:
        height, width = image_shape
    
    # Generate pixel grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    grid_points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
    
    # Interpolate displacement x and y
    dx_grid = griddata(source_points, displacements[:, 0], grid_points, 
                      method=method, fill_value=0).reshape(height, width)
    dy_grid = griddata(source_points, displacements[:, 1], grid_points, 
                      method=method, fill_value=0).reshape(height, width)
    
    # Smooth
    dx_field = gaussian_filter(dx_grid, sigma=sigma)
    dy_field = gaussian_filter(dy_grid, sigma=sigma)
    
    # Constrain magnitude
    magnitude = np.sqrt(dx_field**2 + dy_field**2)
    scale = np.minimum(1.0, max_displacement / (magnitude + 1e-8))
    dx_field *= scale
    dy_field *= scale
    
    return np.stack((dx_field, dy_field), axis=-1)


def convert_4x4_to_3x3_transform(transform_4x4):
    """
    Convert 4x4 transformation matrix to 3x3 affine matrix
    
    Args:
        transform_4x4: 4x4 transformation matrix
        
    Returns:
        3x3 affine transformation matrix
    """
    transform_3x3 = np.eye(3)
    transform_3x3[0:2, 0:2] = transform_4x4[0:2, 0:2]
    transform_3x3[0:2, 2] = transform_4x4[0:2, 3]
    return transform_3x3