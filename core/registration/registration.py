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
import core.registration.rigid as rigid
import pandas as pd


# ---------------------- Rigid Registration ---------------------- #

def perform_rigid_registration(source_prep, target_prep, source_mask, target_mask):
    """
    Perform rigid registration between source and target images.

    Args:
        source_prep (np.ndarray): Preprocessed source image.
        target_prep (np.ndarray): Preprocessed target image.
        source_mask (np.ndarray): Source tissue mask.
        target_mask (np.ndarray): Target tissue mask.

    Returns:
        tuple: (warped_image, transformation_matrix)
    """
    moving_img_transformed, final_transform, _ = rigid.rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )

    height, width = source_prep.shape[:2]
    warped = cv2.warpAffine(source_prep, final_transform[0:-1], (width, height))

    return warped, final_transform


# ---------------------- ICP Registration ---------------------- #

def perform_icp_registration(moving_points, fixed_points, threshold=50000.0):
    """
    Perform ICP registration on 2D point sets.

    Args:
        moving_points (np.ndarray): Moving point set (N, 2).
        fixed_points (np.ndarray): Fixed point set (M, 2).
        threshold (float): Maximum correspondence distance.

    Returns:
        tuple: (transformation_matrix, transformed_points)
    """
    # Convert to Open3D point clouds (z=0 for 2D)
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


# ---------------------- Shape-Aware Registration ---------------------- #

class ShapeAwarePointSetRegistration:
    def __init__(self, fixed_points, moving_points, shape_attribute='area',
                 shape_weight=0.3, max_iterations=50, tolerance=1e-6):
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
        """Estimate similarity transform (rotation, scale, translation) from A → B."""
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

            R_new, t_new, s_new = self._estimate_rigid_transform(moving_xy, matched_fixed)
            R, t, s = R_new, t_new, s_new

            if np.abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        transformed = self._apply_transform(moving_xy, R, t, s)
        self.registered_points = self.moving.copy()
        self.registered_points['registered_x'] = transformed[:, 0]
        self.registered_points['registered_y'] = transformed[:, 1]

        self.rotation = np.arctan2(R[1, 0], R[0, 0])
        self.translation = t
        self.scale = s
        self.final_error = prev_error

        return self.registered_points

    def get_transformation_matrix(self):
        theta = self.rotation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        tx, ty = self.translation
        s = self.scale
        return np.array([
            [s * cos_t, -s * sin_t, tx],
            [s * sin_t,  s * cos_t, ty],
            [0, 0, 1]
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


# ---------------------- Mutual Nearest Neighbors ---------------------- #

def find_mutual_nearest_neighbors(fixed_points, moving_points):
    """
    Find mutual nearest neighbors (MNN) between two point sets.
    """
    nn_fixed_to_moving = NearestNeighbors(n_neighbors=1).fit(moving_points)
    dist1, idx1 = nn_fixed_to_moving.kneighbors(fixed_points)

    nn_moving_to_fixed = NearestNeighbors(n_neighbors=1).fit(fixed_points)
    dist2, idx2 = nn_moving_to_fixed.kneighbors(moving_points)

    mnn_pairs = [(i, j[0]) for i, j in enumerate(idx1) if idx2[j[0]] == i]
    fixed_mnn = np.array([fixed_points[i] for i, _ in mnn_pairs])
    moving_mnn = np.array([moving_points[j] for _, j in mnn_pairs])

    print(f"Matched MNN pairs: {len(mnn_pairs)}")
    return fixed_mnn, moving_mnn, mnn_pairs


# ---------------------- CPD Non-Rigid Registration ---------------------- #

def perform_cpd_registration(moving_points, fixed_points, beta=0.5, alpha=0.01,
                             max_iterations=200, tolerance=1e-9):
    """
    Perform Coherent Point Drift (CPD) non-rigid registration.
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


# ---------------------- Displacement Field ---------------------- #

def create_displacement_field(source_points, transformed_points, image_shape,
                              method='linear', sigma=10.0, max_displacement=10.0):
    """
    Generate dense displacement field from sparse non-rigid registration results.
    """
    displacements = transformed_points - source_points
    height, width = image_shape[:2] if len(image_shape) == 3 else image_shape

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    grid_points = np.vstack((x_coords.ravel(), y_coords.ravel())).T

    dx_grid = griddata(source_points, displacements[:, 0], grid_points, method=method, fill_value=0).reshape(height, width)
    dy_grid = griddata(source_points, displacements[:, 1], grid_points, method=method, fill_value=0).reshape(height, width)

    dx_field = gaussian_filter(dx_grid, sigma=sigma)
    dy_field = gaussian_filter(dy_grid, sigma=sigma)

    magnitude = np.sqrt(dx_field**2 + dy_field**2)
    scale = np.minimum(1.0, max_displacement / (magnitude + 1e-8))
    dx_field *= scale
    dy_field *= scale

    return np.stack((dx_field, dy_field), axis=-1)


# ---------------------- Utility ---------------------- #

def convert_4x4_to_3x3_transform(transform_4x4):
    """
    Convert 4x4 transformation matrix to 3x3 affine matrix.
    """
    transform_3x3 = np.eye(3)
    transform_3x3[0:2, 0:2] = transform_4x4[0:2, 0:2]
    transform_3x3[0:2, 2] = transform_4x4[0:2, 3]
    return transform_3x3
