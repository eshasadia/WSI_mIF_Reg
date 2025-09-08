import os
import glob
import copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from probreg import cpd  # CPD registration

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
    xy = np.vstack([points[:, 0], points[:, 1]])
    kde_scores = gaussian_kde(xy)(xy)
    kde_scores = (kde_scores - kde_scores.min()) / (kde_scores.max() - kde_scores.min())
    return kde_scores

def subsample_points(points, max_points=1000):
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        return points[idx]
    return points

# -----------------------------
# Paths
# -----------------------------
folder_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/thumbnail"
he_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Nuclei"
ome_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/moving_nuclei"
output_dir = "/home/u5552013/Datasets/REACTIVAS/Output_CPD_TRE"
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
    target_points = load_points(he_nuclei_dict[key], ['x', 'y']) / 32
    source_points = load_points(ome_nuclei_dict[key], ['global_x', 'global_y']) / 32

    # Optional KDE weights
    source_kde = KDE_cell_density(source_points)
    target_kde = KDE_cell_density(target_points)

    # Subsample points for faster CPD
    source_sub = subsample_points(source_points, max_points=1000)
    target_sub = subsample_points(target_points, max_points=1000)

    # -----------------------------
    # Choose CPD type: 'rigid' or 'nonrigid'
    # -----------------------------
    cpd_type = 'rigid'  # change to 'rigid' if desired

    if cpd_type == 'rigid':
        tf_param, _, _ = cpd.registration_cpd(
            source_sub, target_sub, maxiter=100, update_scale=False
        )
        transformed_source = tf_param.transform(source_points)
    else:
        # Non-rigid CPD
        nonrigid = cpd.NonRigidCPD(
            source_sub, target_sub, beta=2.0, lamb=3.0
        )
        tf_param, _, _ = nonrigid.register()
        transformed_source = tf_param.transform(source_points)

    # -----------------------------
    # Load landmarks (for TRE computation)
    # -----------------------------
    fixed_lm = load_points(he_csv_dict[key], ['x', 'y']) / 32
    moving_lm_raw = load_points(ome_csv_dict[key], ['x', 'y']) / 32

    # Flip y-axis if needed
    img_path = ome_csv_dict[key].replace("_mIF.csv", "_ome_rgb.png")
    if os.path.exists(img_path):
        img_shape = cv2.imread(img_path).shape
        img_height = img_shape[0]
        moving_lm = np.stack([img_height - moving_lm_raw[:, 1], moving_lm_raw[:, 0]], axis=1)
    else:
        moving_lm = moving_lm_raw

    # Apply CPD transform to landmarks
    moving_lm_transformed = tf_param.transform(moving_lm)

    # Compute TRE
    tree = cKDTree(fixed_lm)
    dists, _ = tree.query(moving_lm_transformed)
    tre = dists.mean()
    print(f"[{key}] TRE: {tre:.2f} pixels")

    # Save transformed points
    transformed_df = pd.DataFrame(transformed_source, columns=['x_trans', 'y_trans'])
    transformed_df.to_csv(os.path.join(output_dir, f"{key}_transformed.csv"), index=False)

    # Plot results
    plt.figure(figsize=(6, 6))
    plt.scatter(source_points[:, 0], source_points[:, 1], s=5, label='MxIF nuclei')
    plt.scatter(target_points[:, 0], target_points[:, 1], s=5, label='HE nuclei')
    plt.scatter(transformed_source[:, 0], transformed_source[:, 1], s=5, label='Transformed MxIF')
    plt.scatter(moving_lm_transformed[:, 0], moving_lm_transformed[:, 1], s=10, c='red', label='Transformed landmarks')
    plt.scatter(fixed_lm[:, 0], fixed_lm[:, 1], s=10, c='lime', label='Fixed landmarks')
    plt.scatter(moving_lm[:, 0], moving_lm[:, 1], s=10, c='blue', label='Original moving landmarks')
    plt.title(f"{key} CPD Alignment + TRE: {tre:.2f}")
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, f"{key}_cpd_tre_plot.png"))
    plt.close()

print("All samples processed with CPD + TRE.")
