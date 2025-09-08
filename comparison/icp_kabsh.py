import os
import glob
import copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

# -----------------------------
# Helper functions
# -----------------------------
def get_prefix(filename):
    import re
    match = re.match(r"(\d+)_", os.path.basename(filename))
    return match.group(1) if match else None

def load_points(csv_path, columns):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    cols = [c for c in df.columns if c.lower() in [x.lower() for x in columns]]
    if len(cols) != len(columns):
        raise ValueError(f"CSV {csv_path} missing required columns {columns}")
    return df[cols].values.astype(np.float32)

def KDE_cell_density(points):
    """Compute normalized KDE density for weighting"""
    xy = np.vstack([points[:,0], points[:,1]])
    kde_scores = gaussian_kde(xy)(xy)
    kde_scores = (kde_scores - kde_scores.min()) / (kde_scores.max() - kde_scores.min())
    return kde_scores

def subsample_points(points, max_points=1000):
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        return points[idx]
    return points

def kabsch_2d(P, Q):
    """Compute 2D rigid transformation P->Q"""
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t

def icp_2d(source, target, max_iter=50, tol=1e-5, subsample=1000):
    """ICP with KDTree + Kabsch"""
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src_sub = subsample_points(src, subsample)
    tgt_sub = subsample_points(tgt, subsample)
    prev_error = float('inf')
    for i in range(max_iter):
        tree = cKDTree(tgt_sub)
        dists, idx = tree.query(src_sub)
        closest = tgt_sub[idx]
        R, t = kabsch_2d(src_sub, closest)
        src_sub = (R @ src_sub.T).T + t
        mean_error = np.mean(dists)
        if abs(prev_error - mean_error) < tol:
            break
        prev_error = mean_error
    transformed_source = (R @ source.T).T + t
    return transformed_source, R, t

# -----------------------------
# Paths
# -----------------------------
folder_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/thumbnail"
he_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Nuclei"
ome_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/moving_nuclei"
output_dir = "/home/u5552013/Datasets/REACTIVAS/Output_ICP_TRE"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Collect files
# -----------------------------
he_csv_all = glob.glob(os.path.join(folder_path, "*_he.csv"))
ome_csv_all = glob.glob(os.path.join(folder_path, "*_mIF.csv"))
he_nuclei_all = glob.glob(os.path.join(he_nuclei_path, "*_HE_*.csv"))
ome_nuclei_all = glob.glob(os.path.join(ome_nuclei_path, "*_ome_*.csv"))

he_csv_dict = {get_prefix(f): f for f in he_csv_all}
ome_csv_dict = {get_prefix(f): f for f in ome_csv_all}
he_nuclei_dict = {get_prefix(f): f for f in he_nuclei_all}
ome_nuclei_dict = {get_prefix(f): f for f in ome_nuclei_all}

common_keys = sorted(set(he_csv_dict) & set(ome_csv_dict) & set(he_nuclei_dict) & set(ome_nuclei_dict))
print(f"Found {len(common_keys)} common samples.")

# -----------------------------
# Main loop
# -----------------------------
for key in common_keys:
    print(f"Processing {key}")

    # Load nuclei points
    source_points = load_points(he_nuclei_dict[key], ['x','y']) / 32
    target_points = load_points(ome_nuclei_dict[key], ['global_x','global_y']) / 32

    # Optional: add KDE weighting as third dimension
    # source_kde = KDE_cell_density(source_points)
    # target_kde = KDE_cell_density(target_points)
    source = source_points
    target = target_points

    # Augment to 3D if needed
    source_aug = np.column_stack([source, np.zeros(len(source))])
    target_aug = np.column_stack([target, np.zeros(len(target))])

    # Subsample
    source_sub = subsample_points(source_aug[:, :2], max_points=1000)
    target_sub = subsample_points(target_aug[:, :2], max_points=1000)

    # ICP alignment
    transformed_source, R_est, t_est = icp_2d(source_sub, target_sub, max_iter=50, subsample=1000)

    # Apply transformation to full source
    full_transformed_source = (R_est @ source.T).T + t_est

    # Load landmarks for TRE
    fixed_lm = load_points(he_csv_dict[key], ['x','y']) / 32
    moving_lm_raw = load_points(ome_csv_dict[key], ['x','y'])
    img_shape = cv2.imread(ome_csv_dict[key].replace("_mIF.csv", "_ome_rgb.png")).shape[:2]
    moving_lm = np.stack([img_shape[0] - moving_lm_raw[:,1], moving_lm_raw[:,0]], axis=1) / 32

    # Apply transformation to landmarks
    moving_lm_transformed = (R_est @ moving_lm.T).T + t_est

    # Compute TRE
    tree = cKDTree(fixed_lm)
    dists, _ = tree.query(moving_lm_transformed)
    tre = dists.mean()
    print(f"[{key}] TRE: {tre:.2f} pixels")

    # Save transformed points
    transformed_df = pd.DataFrame(full_transformed_source, columns=['x_trans','y_trans'])
    transformed_df.to_csv(os.path.join(output_dir, f"{key}_transformed.csv"), index=False)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(source[:,0], source[:,1], s=5, label='HE nuclei')
    plt.scatter(target[:,0], target[:,1], s=5, label='MxIF nuclei')
    plt.scatter(full_transformed_source[:,0], full_transformed_source[:,1], s=5, label='Transformed HE')
    plt.scatter(moving_lm_transformed[:,0], moving_lm_transformed[:,1], s=10, c='red', label='Transformed landmarks')
    plt.scatter(fixed_lm[:,0], fixed_lm[:,1], s=10, c='lime', label='Fixed landmarks')
    plt.title(f"{key} ICP Alignment + TRE: {tre:.2f}")
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, f"{key}_icp_tre_plot.png"))
    plt.close()

print("All samples processed with ICP + TRE.")
