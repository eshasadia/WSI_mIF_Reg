# """
# Nuclei detection and analysis functions
# """

# import cv2
# import numpy as np
# import pandas as pd
# from tiatoolbox.tools import patchextraction
# from tiatoolbox.tools.registration.wsi_registration import AffineWSITransformer
# import wsi_mif_registration.utils.util as util
# from wsi_mif_registration.config import FIXED_THRESHOLD, MOVING_THRESHOLD, MIN_NUCLEI_AREA, GAMMA_CORRECTION
# from wsi_mif_registration.preprocessing.preprocessing import process_nuclei_patch


# def extract_patches_from_wsi(wsi, mask, patch_size=(1000, 1000), stride=(1000, 1000)):
#     """
#     Extract patches from WSI using sliding window
    
#     Args:
#         wsi: WSI reader object
#         mask: Tissue mask
#         patch_size: Size of patches (width, height)
#         stride: Stride for patch extraction
        
#     Returns:
#         Patch extractor object
#     """
#     patch_extractor = patchextraction.get_patch_extractor(
#         input_img=wsi,
#         method_name="slidingwindow",
#         patch_size=patch_size,
#         input_mask=mask,
#         stride=stride,
#         resolution=0
#     )
#     return patch_extractor


# def process_nuclei_in_patches(fixed_patch_extractor, tfm, start_index=0, end_index=None):
#     """
#     Process nuclei detection in image patches
    
#     Args:
#         fixed_patch_extractor: Patch extractor for fixed image
#         tfm: Affine WSI transformer for moving image
#         start_index: Starting patch index
#         end_index: Ending patch index (None for all patches)
        
#     Returns:
#         tuple: (all_fixed_nuclei_data, all_moving_nuclei_data)
#     """
#     if end_index is None:
#         end_index = len(fixed_patch_extractor) - 1
    
#     all_fixed_nuclei_data = []
#     all_moving_nuclei_data = []
    
#     for idx, patch_idx in enumerate(range(start_index, end_index + 1)):
#         try:
#             # Process fixed image patch
#             fixed_nuclei = process_fixed_patch(fixed_patch_extractor, patch_idx)
#             all_fixed_nuclei_data.extend(fixed_nuclei)
            
#             # Process moving image patch
#             moving_nuclei = process_moving_patch(fixed_patch_extractor, tfm, patch_idx)
#             all_moving_nuclei_data.extend(moving_nuclei)
            
#         except Exception as e:
#             print(f"Error processing patch {patch_idx}: {e}")
    
#     return all_fixed_nuclei_data, all_moving_nuclei_data


# def process_fixed_patch(patch_extractor, patch_idx):
#     """
#     Process nuclei detection in a fixed image patch
    
#     Args:
#         patch_extractor: Patch extractor object
#         patch_idx: Index of patch to process
        
#     Returns:
#         List of nuclei data dictionaries
#     """
#     # Get the fixed image patch
#     fixed_img = patch_extractor[patch_idx]
    
#     # Process nuclei
#     _, fixed_stats, fixed_centroids = process_nuclei_patch(
#         fixed_img, FIXED_THRESHOLD, min_area=MIN_NUCLEI_AREA
#     )
    
#     # Get patch coordinates
#     coords = patch_extractor.coordinate_list[patch_idx]
    
#     nuclei_data = []
    
#     # Extract centroids and calculate global coordinates
#     for i in range(1, len(fixed_centroids)):  # Skip background at index 0
#         if fixed_stats[i, cv2.CC_STAT_AREA] > MIN_NUCLEI_AREA:
#             x, y = fixed_centroids[i]
            
#             # Calculate global coordinates
#             global_x = coords[0] + x
#             global_y = coords[1] + y
            
#             nuclei_data.append({
#                 'patch_id': patch_idx,
#                 'patch_x1': coords[0],
#                 'patch_y1': coords[1],
#                 'patch_x2': coords[2],
#                 'patch_y2': coords[3],
#                 'local_x': x,
#                 'local_y': y,
#                 'global_x': global_x,
#                 'global_y': global_y,
#                 'area': fixed_stats[i, cv2.CC_STAT_AREA]
#             })
    
#     return nuclei_data


# def process_moving_patch(patch_extractor, tfm, patch_idx):
#     """
#     Process nuclei detection in a moving image patch
    
#     Args:
#         patch_extractor: Patch extractor object for coordinate reference
#         tfm: Affine WSI transformer for moving image
#         patch_idx: Index of patch to process
        
#     Returns:
#         List of nuclei data dictionaries
#     """
#     # Get patch coordinates
#     coords = patch_extractor.coordinate_list[patch_idx]
#     location = (coords[0], coords[1])
#     size = (coords[2] - coords[0], coords[3] - coords[1])
    
#     # Apply transformation to read corresponding region from moving WSI
#     moving_img = tfm.read_rect(location, size, resolution=40, units="power")
    
#     # Process nuclei with gamma correction
#     _, moving_stats, moving_centroids = process_nuclei_patch(
#         moving_img, MOVING_THRESHOLD, gamma=GAMMA_CORRECTION, min_area=MIN_NUCLEI_AREA
#     )
    
#     nuclei_data = []
    
#     # Extract centroids and calculate global coordinates
#     for i in range(1, len(moving_centroids)):  # Skip background at index 0
#         if moving_stats[i, cv2.CC_STAT_AREA] > MIN_NUCLEI_AREA:
#             x, y = moving_centroids[i]
            
#             # Calculate global coordinates
#             global_x = coords[0] + x
#             global_y = coords[1] + y
            
#             nuclei_data.append({
#                 'patch_id': patch_idx,
#                 'patch_x1': coords[0],
#                 'patch_y1': coords[1],
#                 'patch_x2': coords[2],
#                 'patch_y2': coords[3],
#                 'local_x': x,
#                 'local_y': y,
#                 'global_x': global_x,
#                 'global_y': global_y
#             })
    
#     return nuclei_data


# def save_nuclei_data_to_csv(fixed_nuclei_data, moving_nuclei_data, 
#                            fixed_csv_path, moving_csv_path):
#     """
#     Save nuclei data to CSV files
    
#     Args:
#         fixed_nuclei_data: List of fixed nuclei data
#         moving_nuclei_data: List of moving nuclei data
#         fixed_csv_path: Output path for fixed nuclei CSV
#         moving_csv_path: Output path for moving nuclei CSV
#     """
#     # Convert to DataFrames
#     fixed_nuclei_df = pd.DataFrame(fixed_nuclei_data)
#     moving_nuclei_df = pd.DataFrame(moving_nuclei_data)
    
#     # Save to CSV
#     fixed_nuclei_df.to_csv(fixed_csv_path, index=False)
#     moving_nuclei_df.to_csv(moving_csv_path, index=False)
    
#     print(f"Saved {len(fixed_nuclei_data)} fixed nuclei to {fixed_csv_path}")
#     print(f"Saved {len(moving_nuclei_data)} moving nuclei to {moving_csv_path}")


# def load_nuclei_coordinates(csv_path):
#     """
#     Load nuclei coordinates from CSV file
    
#     Args:
#         csv_path: Path to CSV file
        
#     Returns:
#         DataFrame with nuclei coordinates
#     """
#     df = pd.read_csv(csv_path)
    
#     # Ensure 'area' column exists
#     if 'area' not in df.columns:
#         df['area'] = 1.0
    
#     return df


# def extract_nuclei_points(df, columns=['global_x', 'global_y']):
#     """
#     Extract point coordinates from nuclei DataFrame
    
#     Args:
#         df: Nuclei DataFrame
#         columns: Column names for coordinates
        
#     Returns:
#         NumPy array of points (N, 2)
#     """
#     return df[columns].to_numpy()


# def subsample_nuclei(df, n_samples, random_state=42):
#     """
#     Subsample nuclei from DataFrame
    
#     Args:
#         df: Nuclei DataFrame
#         n_samples: Number of samples to extract
#         random_state: Random seed
        
#     Returns:
#         Subsampled DataFrame
#     """
#     if len(df) <= n_samples:
#         return df
    
#     return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


# def create_nuclei_dataframe_from_points(points, area_values=None):
#     """
#     Create nuclei DataFrame from point coordinates
    
#     Args:
#         points: Point coordinates (N, 2)
#         area_values: Optional area values for each point
        
#     Returns:
#         DataFrame with nuclei data
#     """
#     df = pd.DataFrame(points, columns=['global_x', 'global_y'])
    
#     if area_values is not None:
#         df['area'] = area_values
#     else:
#         df['area'] = 1.0
    
#     return df

"""
Nuclei detection and analysis functions
"""

import cv2
import numpy as np
import pandas as pd
from tiatoolbox.tools import patchextraction
from tiatoolbox.tools.registration.wsi_registration import AffineWSITransformer
import wsi_mif_registration.utils.util as util
from wsi_mif_registration.config import FIXED_THRESHOLD, MOVING_THRESHOLD, MIN_NUCLEI_AREA, GAMMA_CORRECTION

def process_nuclei_patch(img, threshold, gamma=None, min_area=200):
    """
    Process a single patch to detect nuclei
    
    Args:
        img: Input image patch
        threshold: Binary threshold value
        gamma: Gamma correction value (optional)
        min_area: Minimum area for nuclei detection
        
    Returns:
        tuple: (binary_image, stats, centroids)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply gamma correction if specified
    if gamma is not None:
        gray = adjust_gamma(gray, gamma)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    return binary, stats, centroids

def extract_patches_from_wsi(wsi, mask, patch_size=(1000, 1000), stride=(1000, 1000)):
    """
    Extract patches from WSI using sliding window
    
    Args:
        wsi: WSI reader object
        mask: Tissue mask
        patch_size: Size of patches (width, height)
        stride: Stride for patch extraction
        
    Returns:
        Patch extractor object
    """
    patch_extractor = patchextraction.get_patch_extractor(
        input_img=wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=mask,
        stride=stride,
        resolution=0
    )
    return patch_extractor


def process_nuclei_in_patches(fixed_patch_extractor, tfm, start_index=0, end_index=None):
    """
    Process nuclei detection in image patches
    
    Args:
        fixed_patch_extractor: Patch extractor for fixed image
        tfm: Affine WSI transformer for moving image
        start_index: Starting patch index
        end_index: Ending patch index (None for all patches)
        
    Returns:
        tuple: (all_fixed_nuclei_data, all_moving_nuclei_data)
    """
    if end_index is None:
        end_index = len(fixed_patch_extractor) - 1
    
    all_fixed_nuclei_data = []
    all_moving_nuclei_data = []
    
    for idx, patch_idx in enumerate(range(start_index, end_index + 1)):
        try:
            # Process fixed image patch
            fixed_nuclei = process_fixed_patch(fixed_patch_extractor, patch_idx)
            all_fixed_nuclei_data.extend(fixed_nuclei)
            
            # Process moving image patch
            moving_nuclei = process_moving_patch(fixed_patch_extractor, tfm, patch_idx)
            all_moving_nuclei_data.extend(moving_nuclei)
            
        except Exception as e:
            print(f"Error processing patch {patch_idx}: {e}")
    
    return all_fixed_nuclei_data, all_moving_nuclei_data


def process_fixed_patch(patch_extractor, patch_idx):
    """
    Process nuclei detection in a fixed image patch
    
    Args:
        patch_extractor: Patch extractor object
        patch_idx: Index of patch to process
        
    Returns:
        List of nuclei data dictionaries
    """
    # Get the fixed image patch
    fixed_img = patch_extractor[patch_idx]
    
    # Process nuclei
    _, fixed_stats, fixed_centroids = process_nuclei_patch(
        fixed_img, 120, min_area=MIN_NUCLEI_AREA
    )
    
    # Get patch coordinates
    coords = patch_extractor.coordinate_list[patch_idx]
    
    nuclei_data = []
    
    # Extract centroids and calculate global coordinates
    for i in range(1, len(fixed_centroids)):  # Skip background at index 0
        if fixed_stats[i, cv2.CC_STAT_AREA] > MIN_NUCLEI_AREA:
            x, y = fixed_centroids[i]
            
            # Calculate global coordinates
            global_x = coords[0] + x
            global_y = coords[1] + y
            
            nuclei_data.append({
                'patch_id': patch_idx,
                'patch_x1': coords[0],
                'patch_y1': coords[1],
                'patch_x2': coords[2],
                'patch_y2': coords[3],
                'local_x': x,
                'local_y': y,
                'global_x': global_x,
                'global_y': global_y,
                'area': fixed_stats[i, cv2.CC_STAT_AREA]
            })
    
    return nuclei_data


def process_moving_patch(patch_extractor, tfm, patch_idx):
    """
    Process nuclei detection in a moving image patch
    
    Args:
        patch_extractor: Patch extractor object for coordinate reference
        tfm: Affine WSI transformer for moving image
        patch_idx: Index of patch to process
        
    Returns:
        List of nuclei data dictionaries
    """
    # Get patch coordinates
    coords = patch_extractor.coordinate_list[patch_idx]
    location = (coords[0], coords[1])
    size = (coords[2] - coords[0], coords[3] - coords[1])
    
    # Apply transformation to read corresponding region from moving WSI
    moving_img = tfm.read_rect(location, size, resolution=40, units="power")
    
    # Process nuclei with gamma correction
    _, moving_stats, moving_centroids = process_nuclei_patch(
        moving_img, MOVING_THRESHOLD, gamma=GAMMA_CORRECTION, min_area=MIN_NUCLEI_AREA
    )
    
    nuclei_data = []
    
    # Extract centroids and calculate global coordinates
    for i in range(1, len(moving_centroids)):  # Skip background at index 0
        if moving_stats[i, cv2.CC_STAT_AREA] > MIN_NUCLEI_AREA:
            x, y = moving_centroids[i]
            
            # Calculate global coordinates
            global_x = coords[0] + x
            global_y = coords[1] + y
            
            nuclei_data.append({
                'patch_id': patch_idx,
                'patch_x1': coords[0],
                'patch_y1': coords[1],
                'patch_x2': coords[2],
                'patch_y2': coords[3],
                'local_x': x,
                'local_y': y,
                'global_x': global_x,
                'global_y': global_y
            })
    
    return nuclei_data


def save_nuclei_data_to_csv(fixed_nuclei_data, moving_nuclei_data, 
                           fixed_csv_path, moving_csv_path):
    """
    Save nuclei data to CSV files
    
    Args:
        fixed_nuclei_data: List of fixed nuclei data
        moving_nuclei_data: List of moving nuclei data
        fixed_csv_path: Output path for fixed nuclei CSV
        moving_csv_path: Output path for moving nuclei CSV
    """
    # Convert to DataFrames
    fixed_nuclei_df = pd.DataFrame(fixed_nuclei_data)
    moving_nuclei_df = pd.DataFrame(moving_nuclei_data)
    
    # Save to CSV
    fixed_nuclei_df.to_csv(fixed_csv_path, index=False)
    moving_nuclei_df.to_csv(moving_csv_path, index=False)
    
    print(f"Saved {len(fixed_nuclei_data)} fixed nuclei to {fixed_csv_path}")
    print(f"Saved {len(moving_nuclei_data)} moving nuclei to {moving_csv_path}")


def load_nuclei_coordinates(csv_path):
    """
    Load nuclei coordinates from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with nuclei coordinates
    """
    df = pd.read_csv(csv_path)
    
    # Ensure 'area' column exists
    if 'area' not in df.columns:
        df['area'] = 1.0
    
    return df


def extract_nuclei_points(df, columns=['global_x', 'global_y']):
    """
    Extract point coordinates from nuclei DataFrame
    
    Args:
        df: Nuclei DataFrame
        columns: Column names for coordinates
        
    Returns:
        NumPy array of points (N, 2)
    """
    return df[columns].to_numpy()


def subsample_nuclei(df, n_samples, random_state=42):
    """
    Subsample nuclei from DataFrame
    
    Args:
        df: Nuclei DataFrame
        n_samples: Number of samples to extract
        random_state: Random seed
        
    Returns:
        Subsampled DataFrame
    """
    if len(df) <= n_samples:
        return df
    
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def create_nuclei_dataframe_from_points(points, area_values=None):
    """
    Create nuclei DataFrame from point coordinates
    
    Args:
        points: Point coordinates (N, 2)
        area_values: Optional area values for each point
        
    Returns:
        DataFrame with nuclei data
    """
    df = pd.DataFrame(points, columns=['global_x', 'global_y'])
    
    if area_values is not None:
        df['area'] = area_values
    else:
        df['area'] = 1.0
    
    return df