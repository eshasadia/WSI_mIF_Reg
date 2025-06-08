"""
Main script for WSI registration workflow
"""

from params import *
from config import *
from wsi_mif_registration.preprocessing.preprocessing import *
from wsi_mif_registration.registration.registration import *
from wsi_mif_registration.evaluation.evaluation import *
from wsi_mif_registration.visualization.visualization import *
from wsi_mif_registration.preprocessing.nuclei_analysis import *


def main():
    """Main registration workflow"""
    
    print("=== WSI Registration Workflow ===")
    
    # 1. Load WSI images
    print("\n1. Loading WSI images...")
    source_wsi, target_wsi, source, target = load_wsi_images(
        SOURCE_WSI_PATH, TARGET_WSI_PATH, PREPROCESSING_RESOLUTION
    )
    
    # 2. Preprocess images
    print("\n2. Preprocessing images...")
    source_prep, target_prep = preprocess_images(source, target)
    
    # 3. Extract tissue masks
    print("\n3. Extracting tissue masks...")
    source_mask, target_mask = extract_tissue_masks(source_prep, target_prep)
    
    # 4. Perform rigid registration
    print("\n4. Performing rigid registration...")
    moving_img_transformed, final_transform = perform_rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )
    
    # Display transformed image
    visualize_transformed_image(moving_img_transformed)
    
    # 5. Scale transformation for high resolution
    print("\n5. Scaling transformation matrix...")
    transform_40x = scale_transformation_matrix(final_transform, PREPROCESSING_RESOLUTION, REGISTRATION_RESOLUTION)
    
    # 6. Extract and visualize patches
    print("\n6. Extracting patches for visualization...")
    fixed_patch_extractor = extract_patches_from_wsi(
        target_wsi, target_mask, PATCH_SIZE, PATCH_STRIDE
    )
    
    # Visualize a sample patch
    loc = fixed_patch_extractor.coordinate_list[70]
    location = (loc[0], loc[1])
    
    # Extract regions for visualization
    fixed_tile = target_wsi.read_rect(location, VISUALIZATION_SIZE, resolution=40, units="power")
    moving_tile = source_wsi.read_rect(location, VISUALIZATION_SIZE, resolution=40, units="power") 
    
    # Create transformer and extract transformed tile
    tfm = AffineWSITransformer(source_wsi, transform_40x)
    transformed_tile = tfm.read_rect(location=location, size=VISUALIZATION_SIZE, resolution=0, units="level")
    
    # Visualize patches
    visualize_patches(fixed_tile, moving_tile, transformed_tile)
    
    # 7. Evaluate registration with landmarks
    print("\n7. Evaluating registration with landmarks...")
    fixed_points, moving_points = load_landmark_points(FIXED_POINTS_PATH, MOVING_POINTS_PATH)
    
    eval_results = evaluate_registration_tre(
        fixed_points, moving_points, final_transform, target.shape, scale_factor=2
    )
    
    print(f"Initial TRE: {eval_results['tre_initial']:.2f}")
    print(f"Coarse TRE: {eval_results['tre_final']:.2f}")
    print(f"Final rTRE: {eval_results['rtre_mean']:.4f}")
    
    # 8. Process nuclei detection
    print("\n8. Processing nuclei detection...")
    all_fixed_nuclei_data, all_moving_nuclei_data = process_nuclei_in_patches(
        fixed_patch_extractor, tfm
    )
    
    # Save nuclei data
    save_nuclei_data_to_csv(
        all_fixed_nuclei_data, all_moving_nuclei_data,
        FIXED_NUCLEI_CSV, MOVING_NUCLEI_CSV
    )
    
    # 9. Load and visualize nuclei coordinates
    print("\n9. Visualizing nuclei coordinates...")
    setup_bokeh_notebook()
    
    moving_df = load_nuclei_coordinates(MOVING_NUCLEI_CSV)
    fixed_df = load_nuclei_coordinates(FIXED_NUCLEI_CSV)
    
    # Create basic nuclei overlay plot
    plot1 = create_nuclei_overlay_plot(moving_df, fixed_df)
    show_plot(plot1)
    
    # Create detailed plot with color mapping
    plot2 = create_detailed_nuclei_plot_with_colormaps(moving_df, fixed_df)
    show_plot(plot2)
    
    # 10. Perform ICP registration on nuclei
    print("\n10. Performing ICP registration on nuclei...")
    moving_points = extract_nuclei_points(moving_df)
    fixed_points = extract_nuclei_points(fixed_df)
    
    icp_transform, moving_rigid_transformed = perform_icp_registration(
        moving_points, fixed_points, RegistrationParams.ICP_THRESHOLD
    )
    
    # Create DataFrame for rigid registered nuclei
    moving_df_rigid = create_nuclei_dataframe_from_points(moving_rigid_transformed)
    
    # 11. Perform non-rigid registration
    print("\n11. Performing non-rigid registration...")
    
    # Find mutual nearest neighbors
    fixed_mnn, moving_mnn, mnn_pairs = find_mutual_nearest_neighbors(
        fixed_points, moving_rigid_transformed
    )
    
    # Perform CPD registration
    moving_nonrigid_transformed = perform_cpd_registration(
        moving_mnn, fixed_mnn,
        beta=RegistrationParams.CPD_BETA,
        alpha=RegistrationParams.CPD_ALPHA,
        max_iterations=RegistrationParams.CPD_MAX_ITERATIONS,
        tolerance=RegistrationParams.CPD_TOLERANCE)