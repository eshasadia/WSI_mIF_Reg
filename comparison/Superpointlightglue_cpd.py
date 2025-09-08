import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial import cKDTree
from probreg import cpd

# -----------------------------
# Helper functions
# -----------------------------
def get_prefix(filename):
    match = re.match(r"(\d+)_", os.path.basename(filename))
    return match.group(1) if match else None

def compute_tre(fixed_points, moving_points):
    """
    fixed_points: (N_fix,2)
    moving_points: (N_mov,2) -- these will be nearest-neighbored to fixed
    returns mean distance (TRE)
    """
    if len(fixed_points) == 0 or len(moving_points) == 0:
        return np.nan
    tree = cKDTree(fixed_points)
    dists, _ = tree.query(moving_points)
    return dists.mean()

def subsample_points(points, max_points=1000):
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        return points[idx]
    return points

# -----------------------------
# Paths and IO
# -----------------------------
folder_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/thumbnail"
he_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Nuclei"
ome_nuclei_path = "/home/u5552013/Datasets/REACTIVAS/Data/Annotations_mif/moving_nuclei"
output_dir = "/home/u5552013/Datasets/REACTIVAS/Output_Homography_CPD_TRE"
os.makedirs(output_dir, exist_ok=True)

# Collect files
he_images_all = glob.glob(os.path.join(folder_path, "*_HE_*.png"))
ome_images_all = glob.glob(os.path.join(folder_path, "*_ome_rgb.png"))
he_csv_all = glob.glob(os.path.join(folder_path, "*_he.csv"))
ome_csv_all = glob.glob(os.path.join(folder_path, "*_mIF.csv"))
he_nuclei_all = glob.glob(os.path.join(he_nuclei_path, "*_HE_*.csv"))
ome_nuclei_all = glob.glob(os.path.join(ome_nuclei_path, "*_ome_*.csv"))

he_dict = {get_prefix(f): f for f in he_images_all}
ome_dict = {get_prefix(f): f for f in ome_images_all}
he_csv_dict = {get_prefix(f): f for f in he_csv_all}
ome_csv_dict = {get_prefix(f): f for f in ome_csv_all}
he_nuclei_dict = {get_prefix(f): f for f in he_nuclei_all}
ome_nuclei_dict = {get_prefix(f): f for f in ome_nuclei_all}

common_keys = sorted(set(he_dict) & set(ome_dict) & set(he_csv_dict) & set(ome_csv_dict) & set(he_nuclei_dict) & set(ome_nuclei_dict))
print(f"Found {len(common_keys)} common samples.")

# CSV summary
summary_rows = []

# -----------------------------
# Load LightGlue model
# -----------------------------
processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# Main loop
# -----------------------------
for key in common_keys:
    print(f"\nProcessing {key}")
    try:
        he_path = he_dict[key]
        ome_path = ome_dict[key]
        he_csv = he_csv_dict[key]
        ome_csv = ome_csv_dict[key]
        he_nuclei = he_nuclei_dict[key]
        ome_nuclei = ome_nuclei_dict[key]

        # ---- Load images for matching ----
        fixed_pil = Image.open(he_path).convert("RGB")
        moving_pil = Image.open(ome_path).convert("RGB")

        images = [fixed_pil, moving_pil]
        inputs = processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        image_sizes = [[(img.height, img.width) for img in images]]
        results = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
        output = results[0]

        kpts0 = output["keypoints0"].cpu().numpy()  # fixed
        kpts1 = output["keypoints1"].cpu().numpy()  # moving

        if len(kpts0) < 10:
            print(f"[{key}] Not enough matches found ({len(kpts0)}). Skipping.")
            summary_rows.append({'id': key, 'tre_homography': np.nan, 'tre_cpd': np.nan})
            continue

        # ---- Compute Homography (moving -> fixed) ----
        H, mask = cv2.findHomography(kpts1, kpts0, cv2.RANSAC, 5.0)
        if H is None:
            print(f"[{key}] Homography could not be computed. Skipping.")
            summary_rows.append({'id': key, 'tre_homography': np.nan, 'tre_cpd': np.nan})
            continue

        # ---- Load images and csvs ----
        fixed = cv2.imread(he_path)
        moving = cv2.imread(ome_path)
        img_height, img_width = moving.shape[:2]

        fixed_lm = pd.read_csv(he_csv)[['x', 'y']].values.astype(np.float32) / 32.0
        moving_lm_raw = pd.read_csv(ome_csv)[['x', 'y']].values.astype(np.float32) / 32.0

        # Transform moving landmarks to match coordinate convention you used before:
        # you previously did [img_height - y, x]
        moving_lm = np.stack([img_height - moving_lm_raw[:, 1], moving_lm_raw[:, 0]], axis=1).astype(np.float32)

        # Nuclei points
        fixed_nuclei = pd.read_csv(he_nuclei)[['x', 'y']].values.astype(np.float32) / 32.0
        moving_nuclei = pd.read_csv(ome_nuclei)[['global_x', 'global_y']].values.astype(np.float32) / 32.0

        # ---- Apply Homography to moving nuclei & landmarks ----
        # cv2.perspectiveTransform expects shape (N,1,2) float32
        moving_nuclei_hom = cv2.perspectiveTransform(moving_nuclei.reshape(-1,1,2).astype(np.float32), H).reshape(-1,2)
        moving_lm_hom = cv2.perspectiveTransform(moving_lm.reshape(-1,1,2).astype(np.float32), H).reshape(-1,2)

        # ---- TRE after Homography ----
        tre_h = compute_tre(fixed_lm, moving_lm_hom)
        print(f"[{key}] TRE after Homography: {tre_h:.4f} px")

        # ---- CPD refinement (3D rigid, z=0) ----
        # Build 3D arrays (z=0)
        source_3d_full = np.column_stack([moving_nuclei_hom, np.zeros(len(moving_nuclei_hom), dtype=np.float32)])
        target_3d_full = np.column_stack([fixed_nuclei, np.zeros(len(fixed_nuclei), dtype=np.float32)])

        # Subsample for speed, but transform full points later using learned transform
        source_sub = subsample_points(source_3d_full, max_points=500)
        target_sub = subsample_points(target_3d_full, max_points=500)

        try:
            tf_param, _, _ = cpd.registration_cpd(
                source_sub, target_sub, maxiter=1, update_scale=False
            )
        except Exception as e:
            print(f"[{key}] CPD failed: {e}")
            # Save homography-only result and continue
            # Save visual & summary
            tre_cpd = np.nan
            summary_rows.append({'id': key, 'tre_homography': float(tre_h), 'tre_cpd': tre_cpd})
            # Save figures
            fig, axs = plt.subplots(1,3,figsize=(18,6))
            axs[0].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)); axs[0].scatter(fixed_nuclei[:,0], fixed_nuclei[:,1], s=5)
            axs[1].imshow(cv2.cvtColor(moving, cv2.COLOR_BGR2RGB)); axs[1].scatter(moving_nuclei[:,0], moving_nuclei[:,1], s=5)
            transformed = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]))
            axs[2].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
            axs[2].scatter(moving_nuclei_hom[:,0], moving_nuclei_hom[:,1], s=5, c='red')
            axs[2].scatter(fixed_nuclei[:,0], fixed_nuclei[:,1], s=5, c='blue', alpha=0.6)
            plt.suptitle(f"{key} - Homography only (CPD failed)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{key}_homography_only.png"))
            plt.close()
            continue

        # Apply the CPD transform to full set of moving nuclei and moving landmarks
        transformed_source_3d_full = tf_param.transform(source_3d_full)  # (N,3)
        transformed_source_2d_full = transformed_source_3d_full[:, :2]

        # Apply to landmarks (augment then transform)
        moving_lm_3d = np.column_stack([moving_lm_hom, np.zeros(len(moving_lm_hom), dtype=np.float32)])
        moving_lm_cpd_3d = tf_param.transform(moving_lm_3d)
        moving_lm_cpd = moving_lm_cpd_3d[:, :2]

        # ---- TRE after CPD ----
        tre_cpd = compute_tre(fixed_lm, moving_lm_cpd)
        print(f"[{key}] TRE after CPD: {tre_cpd:.4f} px")

        # ---- Save results ----
        # Save transformed nuclei (after CPD) to CSV
        transformed_df = pd.DataFrame(transformed_source_2d_full, columns=['x_trans', 'y_trans'])
        transformed_df.to_csv(os.path.join(output_dir, f"{key}_nuclei_transformed_cpd.csv"), index=False)

        # Save TRE summary row
        summary_rows.append({'id': key, 'tre_homography': float(tre_h), 'tre_cpd': float(tre_cpd)})

        # Save visualization
        fig, axs = plt.subplots(1,3,figsize=(18,6))
        axs[0].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
        axs[0].scatter(fixed_nuclei[:,0], fixed_nuclei[:,1], c='blue', s=5, label='HE nuclei')
        axs[0].scatter(fixed_lm[:,0], fixed_lm[:,1], c='lime', s=10, label='HE landmarks')
        axs[0].set_title("Fixed (HE)")

        axs[1].imshow(cv2.cvtColor(moving, cv2.COLOR_BGR2RGB))
        axs[1].scatter(moving_nuclei[:,0], moving_nuclei[:,1], c='green', s=5, label='OME nuclei (orig)')
        axs[1].scatter(moving_lm[:,0], moving_lm[:,1], c='orange', s=10, label='OME landmarks (orig)')
        axs[1].set_title("Moving (OME)")

        transformed_img = cv2.warpPerspective(moving, H, (fixed.shape[1], fixed.shape[0]))
        axs[2].imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
        axs[2].scatter(moving_nuclei_hom[:,0], moving_nuclei_hom[:,1], c='red', s=5, label='After Homography (nuclei)')
        axs[2].scatter(transformed_source_2d_full[:,0], transformed_source_2d_full[:,1], c='magenta', s=4, label='After CPD (nuclei)')
        axs[2].scatter(moving_lm_hom[:,0], moving_lm_hom[:,1], c='orange', s=20, marker='x', label='LM after Homography')
        axs[2].scatter(moving_lm_cpd[:,0], moving_lm_cpd[:,1], c='red', s=20, marker='+', label='LM after CPD')
        axs[2].scatter(fixed_lm[:,0], fixed_lm[:,1], c='lime', s=20, marker='o', label='Fixed landmarks')
        axs[2].set_title(f"Transformed (H then CPD) | TRE H:{tre_h:.2f} CPD:{tre_cpd:.2f}")

        for ax in axs:
            ax.axis('off')
            ax.legend(loc='upper right', fontsize='small')

        plt.suptitle(f"ID: {key}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_homography_cpd_comparison.png"))
        plt.close()

    except Exception as e:
        print(f"[{key}] Unexpected error: {e}")
        summary_rows.append({'id': key, 'tre_homography': np.nan, 'tre_cpd': np.nan})
        continue

# -----------------------------
# Write summary CSV
# -----------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_dir, "tre_summary_per_sample.csv"), index=False)
print("\nDone. Summary saved to:", os.path.join(output_dir, "tre_summary_per_sample.csv"))
