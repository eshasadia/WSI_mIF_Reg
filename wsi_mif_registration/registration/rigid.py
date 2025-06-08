import numpy as np
import cv2
from skimage import exposure
from skimage import img_as_float
from tiatoolbox.utils.metrics import dice
from wsi_mif_registration.evaluation.evaluation import ngf_metric 
import logging
import sys
from tiatoolbox.tools import patchextraction
from tiatoolbox.tools.registration.wsi_registration import AffineWSITransformer
from skimage.registration import phase_cross_correlation
from accelerated_features.modules.xfeat import XFeat
import wsi_mif_registration.utils.util as util
# Constants
RGB_IMAGE_DIM = 3
BIN_MASK_DIM = 2

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




class Trimorph:
    @staticmethod
    def _check_dims(fixed_img, moving_img, fixed_mask, moving_mask):
        if len(np.unique(fixed_mask)) == 1 or len(np.unique(moving_mask)) == 1:
            raise ValueError("The foreground is missing in the mask.")

        if (
            fixed_img.shape[:2] != fixed_mask.shape
            or moving_img.shape[:2] != moving_mask.shape
        ):
            raise ValueError("Mismatch of shape between image and its corresponding mask.")

        if len(fixed_img.shape) == RGB_IMAGE_DIM:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)

        if len(moving_img.shape) == RGB_IMAGE_DIM:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)

        return fixed_img, moving_img

    @staticmethod
    def compute_center_of_mass(mask):
        moments = cv2.moments(mask)
        x_coord_center = moments["m10"] / moments["m00"]
        y_coord_center = moments["m01"] / moments["m00"]
        return (x_coord_center, y_coord_center)

    @staticmethod
    def apply_affine_transformation(fixed_img, moving_img, transform_initializer):
        return cv2.warpAffine(
            moving_img,
            transform_initializer[0:-1][:],
            fixed_img.shape[:2][::-1],
        )

    def prealignment(
        self,
        fixed_img,
        moving_img,
        fixed_mask,
        moving_mask,
        dice_overlap=0.5,
        rotation_step=10,
    ):
        orig_fixed_img, orig_moving_img = fixed_img.copy(), moving_img.copy()

        if len(fixed_mask.shape) != BIN_MASK_DIM:
            fixed_mask = fixed_mask[:, :, 0]
        if len(moving_mask.shape) != BIN_MASK_DIM:
            moving_mask = moving_mask[:, :, 0]

        fixed_mask = (fixed_mask > 0).astype(np.uint8)
        moving_mask = (moving_mask > 0).astype(np.uint8)

        fixed_img = np.squeeze(fixed_img)
        moving_img = np.squeeze(moving_img)

        fixed_img, moving_img = self._check_dims(fixed_img, moving_img, fixed_mask, moving_mask)

        if rotation_step < 10 or rotation_step > 20:
            raise ValueError("Please select the rotation step between 10 and 20.")

        if not (0 <= dice_overlap <= 1):
            raise ValueError("The dice_overlap should be in between 0 and 1.0.")

        fixed_img = exposure.rescale_intensity(img_as_float(fixed_img), in_range=(0, 1))
        moving_img = exposure.rescale_intensity(img_as_float(moving_img), in_range=(0, 1))

        height = int(np.max((fixed_mask.shape[0], moving_mask.shape[0])))
        width = int(np.max((fixed_mask.shape[1], moving_mask.shape[1])))

        padded_fixed_mask = np.pad(
            fixed_mask,
            pad_width=[(0, height - fixed_mask.shape[0]), (0, width - fixed_mask.shape[1])],
            mode="constant",
        )
        padded_moving_mask = np.pad(
            moving_mask,
            pad_width=[(0, height - moving_mask.shape[0]), (0, width - moving_mask.shape[1])],
            mode="constant",
        )
        dice_before = dice(padded_fixed_mask, padded_moving_mask)

        fixed_com = self.compute_center_of_mass((1 - fixed_img) * fixed_mask)
        moving_com = self.compute_center_of_mass((1 - moving_img) * moving_mask)

        com_transform = np.array(
            [[1, 0, fixed_com[0] - moving_com[0]],
             [0, 1, fixed_com[1] - moving_com[1]],
             [0, 0, 1]]
        )

        origin_transform_com_ = np.array([
            [1, 0, -fixed_com[0]],
            [0, 1, -fixed_com[1]],
            [0, 0, 1]
        ])
        origin_transform_com = np.array([
            [1, 0, fixed_com[0]],
            [0, 1, fixed_com[1]],
            [0, 0, 1]
        ])

        all_dice = []
        all_transform = []

        for angle in np.arange(0, 360, rotation_step).tolist():
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            transform = origin_transform_com @ rotation_matrix @ origin_transform_com_ @ com_transform

            warped_moving_mask = cv2.warpAffine(
                moving_mask,
                transform[0:-1][:],
                fixed_img.shape[:2][::-1],
            )
            dice_com = dice(fixed_mask, warped_moving_mask)

            all_dice.append(dice_com)
            all_transform.append(transform)

        if max(all_dice) >= dice_overlap:
            dice_after = max(all_dice)
            pre_transform = all_transform[all_dice.index(dice_after)]

            moving_img = self.apply_affine_transformation(orig_fixed_img, orig_moving_img, pre_transform)
            moving_mask = self.apply_affine_transformation(fixed_img, moving_mask, pre_transform)

            return pre_transform, moving_img, moving_mask, dice_after

        logger.warning(
            "Not able to find the best transformation for pre-alignment. "
            "Try changing the values for 'dice_overlap' and 'rotation_step'."
        )
        return np.eye(3), orig_moving_img, moving_mask, dice_before



"""
fixed_mask = (fixed_mask > 0).astype(np.uint8)
moving_mask = (moving_mask > 0).astype(np.uint8)

# Create instance of Trimorph
aligner = Trimorph()

# Perform prealignment
transform_matrix, aligned_image, aligned_mask, final_dice = aligner.prealignment(
    fixed_img,
    moving_img,
    fixed_mask,
    moving_mask,
    dice_overlap=0.5,
    rotation_step=10
)

# Show result (for debug)
cv2.imshow("Aligned Image", aligned_image)
cv2.imshow("Aligned Mask", aligned_mask * 255)
print("Final Dice Score:", final_dice)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
class XFeatReg:
    def __init__(self, top_k_matches: int = 8000):
        self.matcher = XFeat()
        self.top_k = top_k_matches

    def warp_corners_and_draw_matches(self, ref_points, dst_points, img1, img2):
        """
        Estimate homography, warp corners of img1 to img2, and visualize matches.
        """
        H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1000, confidence=0.999)

        h, w = img1.shape[:2]
        corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

        if H is None:
            print("Homography computation failed. Using identity matrix.")
            H = np.eye(3, dtype=np.float32)
            mask = np.zeros((len(ref_points), 1), dtype=np.uint8)
        else:
            mask = mask.flatten()

        warped_corners = cv2.perspectiveTransform(corners_img1, H)

        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i - 1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        img_matches = cv2.drawMatches(
            img1, keypoints1, img2_with_corners, keypoints2, matches, None,
            matchColor=(0, 255, 0), flags=2
        )

        return img_matches

    def register(self, im1: np.ndarray, im2: np.ndarray):
        """
        Register im2 to im1 using feature matching and affine transform.
        """
        mkpts_0, mkpts_1 = self.matcher.match_xfeat_star(im2, im1, top_k=self.top_k)

        mkpts_0 = np.array(mkpts_0, dtype=np.float32)
        mkpts_1 = np.array(mkpts_1, dtype=np.float32)

        debug_matches = self.warp_corners_and_draw_matches(mkpts_0, mkpts_1, im2, im1)

        M1, mask = cv2.estimateAffinePartial2D(mkpts_0, mkpts_1, method=cv2.RANSAC)
        aligned_img = cv2.warpAffine(im2, M1, (im1.shape[1], im1.shape[0]))

        return M1, aligned_img
    
"""

# Load images
im1 = cv2.imread("/path/to/reference_image.png")
im2 = cv2.imread("/path/to/moving_image.png")

# Create aligner and register images
aligner =  XFeatReg()
M, aligned_image, match_vis = aligner.register(im1, im2)

# Display results
cv2.imshow("Aligned Image", aligned_image)
cv2.imshow("Matches", match_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""


def rigid_registration( moving_img, fixed_img,moving_mask, fixed_mask, verbose=False):
    """
    Aligns a moving image to a fixed image using Trimorph and optionally XFeatReg if needed.
    Args:
        fixed_img (np.ndarray): Fixed image.
        moving_img (np.ndarray): Moving image.
        fixed_mask (np.ndarray): Fixed mask.
        moving_mask (np.ndarray): Moving mask.
        verbose (bool): If True, print intermediate steps.
    Returns:
        moving_img_transformed (np.ndarray): Final aligned moving image.
        final_transform (np.ndarray): Final transformation matrix (3x3).
        timing (dict): Dictionary with timing info.
        ngf_metrics (dict): Dictionary with NGF metrics before and after optional second registration.
    """
    # Make sure you have this module
    ngf_metrics = {}
    # Initial registration with Trimorph
    aligner = Trimorph()
    best_transform1, moving_img_transformed, moving_mask_transformed, max_dice = aligner.prealignment(
        fixed_img, moving_img, fixed_mask, moving_mask
    )
    # Evaluate NGF metric
    ngf_initial = ngf_metric(fixed_img, moving_img_transformed)
    ngf_metrics['NGF trimoph'] = ngf_initial
    # If no transformation was found (identity), do extra registration
    if np.array_equal(best_transform1, np.eye(3)):
        if verbose:
            print("Running extra registration with XFeatReg...")
        aligner = XFeatReg()
        best_transform1, moving_img_transformed = aligner.register(fixed_img, moving_img)
        best_transform1 = np.vstack((best_transform1, [0, 0, 1]))
        ngf_metrics['NGF XFeat'] = ngf_metric(fixed_img, moving_img_transformed)
        final_transform = best_transform1
    else:
      # Optionally perform another registration step with XFeatReg
        aligner = XFeatReg()

        # Register using already transformed image
        M1, aligned_image = aligner.register(fixed_img, moving_img_transformed)
        ngf_second = ngf_metric(fixed_img, aligned_image)
        ngf_metrics['NGF XFeat transformed'] = ngf_second

        # Register directly from original moving image
        M2, third_aligned_image = aligner.register(fixed_img, moving_img)
        ngf_third = ngf_metric(fixed_img, third_aligned_image)
        ngf_metrics['NGF XFeat'] = ngf_third

        # Ensure M1 and M2 are 3x3 transformation matrices (homogeneous coordinates)
        M1_hom = np.vstack([M1, [0, 0, 1]]) if M1.shape == (2, 3) else M1
        M2_hom = np.vstack([M2, [0, 0, 1]]) if M2.shape == (2, 3) else M2

        # Choose the best transformation based on NGF metrics
        if ngf_initial >= ngf_second and ngf_initial >= ngf_third:
            final_transform = best_transform1
        elif ngf_second >= ngf_third:
            final_transform = best_transform1 @ M1_hom
            moving_img_transformed = aligned_image
        else:
            final_transform = M2_hom
            moving_img_transformed = third_aligned_image

        return moving_img_transformed, final_transform, ngf_metrics

def register_wsi_with_fine_tuning(source_wsi, target_wsi, source_mask=None, target_mask=None, initial_transform=None, patch_index=10, patch_size=(5000, 5000), stride=(5000, 5000), base_resolution=0, target_resolution=10, scale_factor=0.625):
    """
    Registers a source WSI to a target WSI using an initial transform and fine-tunes the registration 
    using phase cross-correlation on a specific patch.
    
    Parameters:
    -----------
    source_wsi : object or str
        Source WSI object or path to source WSI file
    target_wsi : object or str
        Target WSI object or path to target WSI file
    source_mask : ndarray, optional
        Mask for the source WSI
    target_mask : ndarray, optional
        Mask for the target WSI
    initial_transform : ndarray, optional
        Initial homogeneous transformation matrix to apply
    patch_index : int, default=10
        Index of the patch to use for fine-tuning
    patch_size : tuple, default=(5000, 5000)
        Size of patches to extract
    stride : tuple, default=(5000, 5000)
        Stride for sliding window patch extraction
    base_resolution : int, default=0
        Base resolution for patch extraction
    target_resolution : int, default=10
        Target resolution for transformation
    scale_factor : float, default=0.625
        Scale factor to apply to the transformation matrix
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'final_transform': The final transformation matrix
        - 'registered_image': The registered image patch
        - 'fixed_tile': The fixed (target) image patch
        - 'transformed_tile': The transformed (source) image patch before fine-tuning
        - 'shift': The calculated shift between patches
    """
   
    
    # Set up patch extractors for both target and source WSIs
    fixed_patch_extractor = patchextraction.get_patch_extractor(
        input_img=target_wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=target_mask,
        stride=stride,
        resolution=base_resolution
    )
    
    moving_patch_extractor = patchextraction.get_patch_extractor(
        input_img=source_wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=source_mask,
        stride=stride,
        resolution=base_resolution
    )
    
    # Use the provided initial transform or identity matrix
    M_homogeneous = initial_transform if initial_transform is not None else np.eye(3)
    
    # Scale the transformation matrix for the target resolution
    transform_40x = util.scale_transformation_matrix(M_homogeneous, scale_factor, target_resolution)
    
    # Get the fixed tile from the target WSI
    fixed_tile = fixed_patch_extractor[patch_index]
    
    # Create transformer for the source WSI
    tfm = AffineWSITransformer(source_wsi, transform_40x)
    
    # Get coordinates of the selected patch
    coords = fixed_patch_extractor.coordinate_list[patch_index]
    location = (coords[0], coords[1])
    size = (coords[2] - coords[0], coords[3] - coords[1])
    
    # Read the corresponding patch from the transformed source WSI
    transformed_tile = tfm.read_rect(location, size, resolution=target_resolution, units="power")
    
    # Optional gamma correction can be uncommented if needed
    # transformed_tile = util.gamma_corrections(transformed_tile, 0.8)
    
    # Calculate shift between the fixed and transformed tiles using phase cross-correlation
    shift, error, diffphase = phase_cross_correlation(fixed_tile, transformed_tile)
    
    # Create translation matrix based on the calculated shift
    translation_offset = np.array([
        [1, 0, -shift[1]], 
        [0, 1, -shift[0]], 
        [0, 0, 1]
    ])
    
    # Apply the translation to get the registered image
    registered_image = cv2.warpAffine(
        transformed_tile, 
        translation_offset[0:-1], 
        (transformed_tile.shape[1], transformed_tile.shape[0])
    )
    
    # Update the transformation matrix with the fine-tuned translation
    transform_40x[0, 2] += translation_offset[0, 2]  # Add x translation
    transform_40x[1, 2] += translation_offset[1, 2]  # Add y translation
    
    # Recalculate the transformer with the updated transformation matrix
    updated_tfm = AffineWSITransformer(source_wsi, transform_40x)
    
    # Extract the tiles again using the updated transformer
    fixed_tile_updated = fixed_patch_extractor[patch_index]
    
    # Get the updated transformed tile using the updated transformer
    updated_transformed_tile = updated_tfm.read_rect(location, size, resolution=target_resolution, units="power")
    
    return {
        'final_transform': transform_40x,
        'registered_image': registered_image,
        'original_fixed_tile': fixed_tile,
        'original_transformed_tile': transformed_tile,
        'updated_fixed_tile': fixed_tile_updated,
        'updated_transformed_tile': updated_transformed_tile,
        'shift': shift,
        'updated_transformer': updated_tfm
    }
def extract_corresponding_tiles(source_wsi, target_wsi, source_mask=None, target_mask=None, 
                               transform_matrix=None, patch_index=10, patch_size=(5000, 5000), 
                               stride=(5000, 5000), base_resolution=0, target_resolution=10, 
                               scale_factor=0.625):
    """
    Extracts corresponding tiles from source and target WSIs using a transformation matrix.
    
    Parameters:
    -----------
    source_wsi : object or str
        Source WSI object or path to source WSI file
    target_wsi : object or str
        Target WSI object or path to target WSI file
    source_mask : ndarray, optional
        Mask for the source WSI
    target_mask : ndarray, optional
        Mask for the target WSI
    transform_matrix : ndarray, optional
        Homogeneous transformation matrix to apply
    patch_index : int, default=10
        Index of the patch to extract
    patch_size : tuple, default=(5000, 5000)
        Size of patches to extract
    stride : tuple, default=(5000, 5000)
        Stride for sliding window patch extraction
    base_resolution : int, default=0
        Base resolution for patch extraction
    target_resolution : int, default=10
        Target resolution for transformation
    scale_factor : float, default=0.625
        Scale factor to apply to the transformation matrix
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'fixed_tile': The fixed (target) image patch
        - 'transformed_tile': The transformed (source) image patch
        - 'transform_40x': The scaled transformation matrix
        - 'location': The location of the extracted patch
        - 'size': The size of the extracted patch
        - 'patch_extractor_fixed': The patch extractor for the target WSI
        - 'patch_extractor_moving': The patch extractor for the source WSI
        - 'transformer': The AffineWSITransformer used
    """
  
    
    # Set up patch extractors for both target and source WSIs
    fixed_patch_extractor = patchextraction.get_patch_extractor(
        input_img=target_wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=target_mask,
        stride=stride,
        resolution=base_resolution
    )
    
    moving_patch_extractor = patchextraction.get_patch_extractor(
        input_img=source_wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=source_mask,
        stride=stride,
        resolution=base_resolution
    )
    
    # Use the provided transform matrix or identity matrix
    M_homogeneous = transform_matrix if transform_matrix is not None else np.eye(3)
    
    # Scale the transformation matrix for the target resolution
    transform_40x = util.scale_transformation_matrix(M_homogeneous, scale_factor, target_resolution)
    
    # Get the fixed tile from the target WSI
    fixed_tile = fixed_patch_extractor[patch_index]
    
    # Create transformer for the source WSI
    tfm = AffineWSITransformer(source_wsi, transform_40x)
    
    # Get coordinates of the selected patch
    coords = fixed_patch_extractor.coordinate_list[patch_index]
    location = (coords[0], coords[1])
    size = (coords[2] - coords[0], coords[3] - coords[1])
    
    # Read the corresponding patch from the transformed source WSI
    transformed_tile = tfm.read_rect(location, size, resolution=target_resolution, units="power")
    
    return {
        'fixed_tile': fixed_tile,
        'transformed_tile': transformed_tile,
        'transform_40x': transform_40x,
    }