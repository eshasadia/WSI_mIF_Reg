import torch as t
import torch.nn.functional as tfun
import torchvision.transforms as trans
from typing import Callable, Optional, Tuple, List, Union, Dict, Any
import torch.optim as topt
import numpy as np
import math
import cv2
from skimage import color
import SimpleITK as sitk
import PIL
import os
from pathlib import Path
from tqdm.auto import tqdm  # Added progress bar support
import logging  # Added logging support
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
import torch as tc
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("elastic_registration")


def build_reference_coordinate_system(input_tensor: Optional[t.Tensor] = None,
                                   dimensions: Optional[t.Size] = None,
                                   compute_device: Optional[Union[str, t.device]] = None) -> t.Tensor:
 
    if input_tensor is not None:
        dimensions = input_tensor.size()
    
    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = t.device(compute_device)
    
    if compute_device is None and input_tensor is not None:
        base_transform = t.eye(len(dimensions)-1)[:-1, :].unsqueeze(0).type_as(input_tensor)
    else:
        base_transform = t.eye(len(dimensions)-1, device=compute_device)[:-1, :].unsqueeze(0)
    
    base_transform = t.repeat_interleave(base_transform, dimensions[0], dim=0)
    coordinate_grid = tfun.affine_grid(base_transform, dimensions, align_corners=False)
    
    return coordinate_grid


def smooth_with_gaussian_kernel(input_tensor: t.Tensor, blur_sigma: float) -> t.Tensor:

    with t.set_grad_enabled(False):
        kernel_width = int(blur_sigma * 2.54) + 1 
        if kernel_width % 2 == 0:
            kernel_width += 1
        return trans.GaussianBlur(kernel_width, blur_sigma)(input_tensor)


def calculate_smoothness_penalty(vector_field: t.Tensor, 
                               compute_device: t.device = t.device("cuda"),
                               weight_map: Optional[t.Tensor] = None) -> t.Tensor:
    dim_count = len(vector_field.size()) - 2
    
    if dim_count == 2:
        x_grad = ((vector_field[:, 1:, :, :] - vector_field[:, :-1, :, :]) * 
              vector_field.shape[1])**2
        y_grad = ((vector_field[:, :, 1:, :] - vector_field[:, :, :-1, :]) * 
              vector_field.shape[2])**2
        
        if weight_map is not None:
            # Apply spatial weighting if provided
            x_weight = weight_map[:, 1:, :].unsqueeze(-1)
            y_weight = weight_map[:, :, 1:].unsqueeze(-1)
            smoothness_term = (t.mean(x_grad * x_weight) + t.mean(y_grad * y_weight)) / 2
        else:
            smoothness_term = (t.mean(x_grad) + t.mean(y_grad)) / 2
    else:
        raise ValueError("Unsupported dimensionality. Must be 2D or 3D.")
        
    return smoothness_term


def initialize_zero_vector_field(input_tensor: t.Tensor) -> t.Tensor:
 
    dim_count = len(input_tensor.size()) - 2
    return t.zeros((input_tensor.size(0), input_tensor.size(2), input_tensor.size(3)) + (dim_count,)).type_as(input_tensor)


def scale_tensor_to_dimensions(input_tensor: t.Tensor, 
                            target_dimensions: t.Size, 
                            interpolation_method: str = 'bilinear') -> t.Tensor:

    return tfun.interpolate(input_tensor, size=target_dimensions, 
                          mode=interpolation_method, align_corners=False)



def compute_normalized_cross_correlation(sources: t.Tensor, 
                                      targets: t.Tensor, 
                                      device: Optional[Union[str, t.device]] = None, 
                                      **config_params) -> t.Tensor:
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size =7
    except:
        win_size = 3
   
    window = (win_size, ) * ndim
    if device is None:
        sum_filt = tc.ones([1, 1, *window]).type_as(sources)
    else:
        sum_filt = tc.ones([1, 1, *window], device=device)



    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)


def apply_deformation_field(input_tensor: t.Tensor, 
                         vector_field: t.Tensor, 
                         coord_grid: Optional[t.Tensor] = None, 
                         interpolation_method: str = 'bilinear', 
                         boundary_handling: str = 'zeros', 
                         compute_device: Optional[Union[str, t.device]] = None) -> t.Tensor:

    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = t.device(compute_device)
        
    if coord_grid is None:
        coord_grid = build_reference_coordinate_system(input_tensor=input_tensor, compute_device=compute_device)
        
    sampling_coordinates = coord_grid + vector_field
    deformed_tensor = tfun.grid_sample(input_tensor, sampling_coordinates, 
                                    mode=interpolation_method, 
                                    padding_mode=boundary_handling, 
                                    align_corners=False)
    
    return deformed_tensor


def scale_deformation_field(vector_field: t.Tensor, 
                          new_dimensions: Union[t.Size, Tuple[int, int]], 
                          interpolation_method: str = 'bilinear') -> t.Tensor:

    # Permute to channel-first format for interpolation
    channel_first = vector_field.permute(0, 3, 1, 2)
    
    # Perform interpolation
    resized = tfun.interpolate(
        channel_first, 
        size=new_dimensions, 
        mode=interpolation_method, 
        align_corners=False
    )
    
    # Return to original format
    return resized.permute(0, 2, 3, 1)


def create_multiscale_representation(input_tensor: t.Tensor, 
                                   level_count: int, 
                                   interpolation_method: str = 'bilinear',
                                   scale_factor: float = 2.0) -> List[t.Tensor]:
  
    pyramid_levels = [None] * level_count
    
    # Build from fine to coarse
    for i in range(level_count - 1, -1, -1):
        if i == level_count - 1:
            # Original resolution
            pyramid_levels[i] = input_tensor
        else:
            # Get previous level and compute dimensions for current level
            prev_size = pyramid_levels[i+1].size()
            current_dims = tuple(int(prev_size[j] / scale_factor) if j > 1 else prev_size[j] 
                               for j in range(len(prev_size)))
            
            # Extract just the spatial dimensions
            spatial_dims = t.Size(current_dims)[2:]
            
            # Apply smoothing to prevent aliasing, then downsample
            smoothed = smooth_with_gaussian_kernel(pyramid_levels[i+1], 1)
            downsampled = scale_tensor_to_dimensions(smoothed, spatial_dims, 
                                                 interpolation_method)
            
            pyramid_levels[i] = downsampled
            
    return pyramid_levels


def convert_image_to_tensor(img_array: np.ndarray, compute_device: Union[str, t.device] = "cpu") -> t.Tensor:
  
    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = t.device(compute_device)
        
    # Normalize image if it's not already in [0, 1] range
    if img_array.dtype != np.float32 and img_array.dtype != np.float64:
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
    
    if len(img_array.shape) == 3:
        # Color image
        return t.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(compute_device)
    elif len(img_array.shape) == 2:
        # Grayscale image
        return t.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(compute_device)
    else:
        raise ValueError(f"Unsupported image dimensions: {img_array.shape}")


def prepare_image_tensors(source_image: np.ndarray, 
                        target_image: np.ndarray, 
                        compute_device: Union[str, t.device],
                        normalize: bool = True) -> Tuple[t.Tensor, t.Tensor]:

    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = t.device(compute_device)
        
    # Convert to grayscale if RGB
    if len(source_image.shape) == 3 and source_image.shape[2] == 3:
        gray_source = color.rgb2gray(source_image)
    else:
        gray_source = source_image
        
    if len(target_image.shape) == 3 and target_image.shape[2] == 3:
        gray_target = color.rgb2gray(target_image)
    else:
        gray_target = target_image

    # Normalize if requested
    if normalize:
        gray_source = (gray_source - gray_source.min()) / (gray_source.max() - gray_source.min() + 1e-10)
        gray_target = (gray_target - gray_target.min()) / (gray_target.max() - gray_target.min() + 1e-10)

    # Convert to tensor format
    tensor_source = convert_image_to_tensor(gray_source, compute_device)
    tensor_target = convert_image_to_tensor(gray_target, compute_device)

    # Create tensors with gradient tracking
    source_tensor = t.tensor(tensor_source, dtype=t.float32, requires_grad=True).to(compute_device)
    target_tensor = t.tensor(tensor_target, dtype=t.float32, requires_grad=True).to(compute_device)

    return source_tensor, target_tensor


def transform_matrix_to_deformation_field(transform_matrix: t.Tensor, 
                                        output_dimensions: t.Size) -> t.Tensor:
  
    # Generate deformation field from matrix
    deformation_grid = tfun.affine_grid(transform_matrix, 
                                      size=output_dimensions, 
                                      align_corners=False)
    
    # Create identity grid to compute displacement
    grid_dimensions = (deformation_grid.size(0), 1) + deformation_grid.size()[1:-1]
    identity_grid = build_reference_coordinate_system(
        dimensions=grid_dimensions, 
        compute_device=transform_matrix.device
    )
    
    # Displacement is the difference from identity
    displacement_vectors = deformation_grid - identity_grid
    
    return displacement_vectors





def tv_regularizer(vector_field: t.Tensor,
                           compute_device: t.device = t.device("cuda"),
                           **config_params) -> t.Tensor:
    """Total Variation regularizer."""
    isotropic = config_params.get('isotropic', False)
    
    dim_count = len(vector_field.size()) - 2
    
    if dim_count == 2:
        dx = vector_field[:, 1:, :, :] - vector_field[:, :-1, :, :]
        dy = vector_field[:, :, 1:, :] - vector_field[:, :, :-1, :]
        
        if isotropic:
            # Isotropic TV: L2 norm over displacement components, L1 over spatial
            grad_magnitude = t.sqrt(t.sum(dx**2, dim=-1) + t.sum(dy**2, dim=-1) + 1e-8)
            tv_loss = t.mean(grad_magnitude)
        else:
            # Anisotropic TV: L1 norm everywhere
            tv_loss = t.mean(t.abs(dx)) + t.mean(t.abs(dy))
    else:
        raise ValueError("3D TV not implemented yet")
    
    return tv_loss





def diffusion_adaptive_tc(displacement_field: tc.Tensor, compute_device: str = "cuda", **params):
    """Adaptive diffusion regularization."""
    lambda_0 = params.get('lambda_0', 1.0)
    edge_threshold = params.get('edge_threshold', 0.1)
    
    ndim = len(displacement_field.size()) - 2
    
    if ndim == 2:
        # Compute gradients
        dx = displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :]  # [B, H-1, W, 2]
        dy = displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :]  # [B, H, W-1, 2]
        
        # Compute gradient magnitudes for each component separately
        grad_mag_x = tc.sqrt(tc.sum(dx**2, dim=-1) + 1e-8)  # [B, H-1, W]
        grad_mag_y = tc.sqrt(tc.sum(dy**2, dim=-1) + 1e-8)  # [B, H, W-1]
        
        # Compute adaptive weights
        weights_x = lambda_0 * tc.exp(-(grad_mag_x**2) / (edge_threshold**2))  # [B, H-1, W]
        weights_y = lambda_0 * tc.exp(-(grad_mag_y**2) / (edge_threshold**2))  # [B, H, W-1]
        
        # Apply weights to squared gradients (sum over displacement components)
        weighted_dx = weights_x * tc.sum(dx**2, dim=-1)  # [B, H-1, W]
        weighted_dy = weights_y * tc.sum(dy**2, dim=-1)  # [B, H, W-1]
        
        # Compute final regularization term
        diffusion_reg = (tc.mean(weighted_dx) + tc.mean(weighted_dy)) / 2
        
    return diffusion_reg



def elastic_image_registration(
    moving_img: np.ndarray, 
    fixed_img: np.ndarray, 
    similarity_metric: str = "ncc",  # Added option to select similarity metric
    similarity_metric_params: Dict[str, Any] = {}, 
    deformation_regularization_params: Dict[str, Any] = {},
    constraint_function: Optional[Callable] = None, 
    constraint_params: Dict[str, Any] = {},
    starting_deformation: Optional[t.Tensor] = None,
    compute_device: Union[str, t.device] = "cuda", 
    verbose: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    save_intermediate: bool = False
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Performs  elastic image registration.
    
    Args:
        
    Returns:
        Tuple of (final_deformation_field, warped_source_image)
    """
    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = t.device(compute_device)
        
    logger.info(f"Using device: {compute_device}")
    
    # Store original images
    original_moving, original_fixed = moving_img, fixed_img
    
    # Prepare tensors for registration
    original_moving_tensor, original_fixed_tensor = prepare_image_tensors(
        original_moving, original_fixed, compute_device
    )
    
    # Apply pre-alignment to ensure images have same dimensions
    aligned_moving = cv2.warpAffine(
        moving_img, 
        np.eye(2, 3), 
        (fixed_img.shape[1], fixed_img.shape[0]),
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Convert to tensors for registration
    moving_tensor, fixed_tensor = prepare_image_tensors(aligned_moving, fixed_img, compute_device)
    
    logger.info(f"Moving image tensor size: {moving_tensor.size()}")
    pyramid_levels=7
    # Create multi-resolution pyramids
    logger.info(f"Creating {pyramid_levels}-level image pyramids")
    moving_pyramid = create_multiscale_representation(moving_tensor, level_count=pyramid_levels)
    fixed_pyramid = create_multiscale_representation(fixed_tensor, level_count=pyramid_levels)
    
    # Create output directory if needed
    if save_intermediate and output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose similarity metric function
    if similarity_metric.lower() == "ncc":
        similarity_function = compute_normalized_cross_correlation
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    active_levels=7
    learning_rates_per_level=[0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    # Process each resolution level
    for level_idx in range(active_levels):
        # Get images at current resolution
        current_moving = moving_pyramid[level_idx]
        current_fixed = fixed_pyramid[level_idx]
        
        logger.info(f"Starting level {level_idx}/{active_levels-1} - " +
                    f"Resolution: {current_moving.size(2)}x{current_moving.size(3)}")
        
        # Initialize or resize deformation field for current level
        if level_idx == 0:
            if starting_deformation is None:
                logger.info("Initializing zero displacement field")
                deformation_field = initialize_zero_vector_field(current_moving).detach().clone()
                deformation_field.requires_grad = True
            else:
                logger.info("Using provided initial displacement field")
                deformation_field = scale_deformation_field(
                    starting_deformation, 
                    (current_moving.size(2), current_moving.size(3))
                ).detach().clone()
                deformation_field.requires_grad = True
                
            # Initialize optimizer
            optimizer = topt.Adam([deformation_field], learning_rates_per_level[level_idx])
        else:
            # Upscale previous field to current resolution
            logger.info(f"Upscaling deformation field to {current_moving.size(2)}x{current_moving.size(3)}")
            deformation_field = scale_deformation_field(
                deformation_field, 
                (current_moving.size(2), current_moving.size(3))
            ).detach().clone()
            deformation_field.requires_grad = True
            
            # Reset optimizer for new level
            optimizer = topt.Adam([deformation_field], learning_rates_per_level[level_idx])
        iterations_per_level=7*[400]
        regularization_weights=[12, 1.2, 1.2, 1.2, 1.2, 0.10, 0.6]
        # regularization_weights=1
        # Optimization at current level
        pbar = tqdm(range(iterations_per_level[level_idx]), disable=not verbose)
        for iter_idx in pbar:
            with t.set_grad_enabled(True):
                # Apply current deformation
                warped_moving = apply_deformation_field(current_moving, deformation_field, 
                                                    compute_device=compute_device)
                
                # Print initial metrics
                if iter_idx == 0 and verbose:
                    initial_similarity = similarity_function(
                        current_moving, current_fixed, 
                        compute_device=compute_device, 
                        **similarity_metric_params
                    )
                    first_warp_similarity = similarity_function(
                        warped_moving, current_fixed, 
                        compute_device=compute_device, 
                        **similarity_metric_params
                    )
                    logger.info(f"Initial similarity: {initial_similarity.item():.6f}")
                    logger.info(f"First warp similarity: {first_warp_similarity.item():.6f}")
                
                # Calculate loss components
                similarity_loss = similarity_function(
                    warped_moving, current_fixed, 
                    compute_device=compute_device, 
                    **similarity_metric_params
                )
                # smoothness_loss_adapt = diffusion_adaptive_tc(
                #     deformation_field, 
                #     compute_device=compute_device, 
                #     **deformation_regularization_params
                # )
                # print("adaptive diffusive regularizer",smoothness_loss_adapt)
                # smoothness_loss_tv = tv_regularizer(
                #     deformation_field, 
                #     compute_device=compute_device, 
                #     **deformation_regularization_params
                # )
                # print("tv regularizer",smoothness_loss_tv)
                smoothness_loss = calculate_smoothness_penalty(
                    deformation_field, 
                    compute_device=compute_device, 
                    **deformation_regularization_params
                )
                # print("relative diffusive regularizer",smoothness_loss_adapt)
                # smoothness_loss=max(smoothness_loss_adapt,smoothness_loss_reg, smoothness_loss_tv)
                # Combine with regularization weight
                total_loss = similarity_loss + regularization_weights[level_idx] * smoothness_loss
                
                # Add optional constraint if provided
                if constraint_function is not None:
                    constraint_loss = constraint_function(deformation_field, **constraint_params)
                    total_loss = total_loss + constraint_loss
                
                # Backpropagate and update
                total_loss.backward()
                optimizer.step()
                
            # Reset gradient for next iteration
            optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_description(
                f"Level {level_idx}/{active_levels-1} | "
                f"Sim: {similarity_loss.item():.4f} | "
                f"Smooth: {smoothness_loss.item():.4f}"
            )
            
            # Save intermediate results if requested
            if save_intermediate and output_dir is not None and iter_idx % 25 == 0:
                # Convert warped image to numpy
                warped_np = warped_moving.detach().cpu().numpy()[0, 0]
                
                # Save warped image
                img_path = output_dir / f"level{level_idx}_iter{iter_idx}_warped.png"
                cv2.imwrite(str(img_path), (warped_np * 255).astype(np.uint8))
    
    # If not all levels were used, upscale to full resolution
    if active_levels != pyramid_levels:
        logger.info(f"Upscaling final field to full resolution {original_moving_tensor.size(2)}x{original_moving_tensor.size(3)}")
        deformation_field = scale_deformation_field(
            deformation_field, 
            (original_moving_tensor.size(2), original_moving_tensor.size(3))
        )
    
    # Report final deformation field statistics
    logger.info(f"Final deformation field range: {deformation_field.min().item():.4f} to {deformation_field.max().item():.4f}")
    
    # Apply final deformation to source image
    final_warped = apply_deformation_field(original_moving_tensor, deformation_field)
    
    # Compute final similarity
    final_similarity = similarity_function(
        final_warped, original_fixed_tensor,
        compute_device=compute_device,
        **similarity_metric_params
    )
    logger.info(f"Final similarity score: {final_similarity.item():.6f}")
    
    # Save final results if requested
    if output_dir is not None:
        # Save final warped image
        final_warped_np = final_warped.detach().cpu().numpy()[0, 0]
        final_img_path = output_dir / "final_warped.png"
        cv2.imwrite(str(final_img_path), (final_warped_np * 255).astype(np.uint8))
        
        # Create comparison image (side by side)
        fixed_np = original_fixed_tensor.detach().cpu().numpy()[0, 0]
        comparison = np.hstack((fixed_np, final_warped_np))
        comparison_path = output_dir / "comparison.png"
        cv2.imwrite(str(comparison_path), (comparison * 255).astype(np.uint8))
    
    return deformation_field, final_warped