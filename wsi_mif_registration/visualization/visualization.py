"""
Visualization functions for WSI registration results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, Title
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256, Inferno256
from bokeh.io import output_notebook
import sys
import importlib
from wsi_mif_registration.config import VisualizationParams


def setup_bokeh_notebook():
    """Setup Bokeh for notebook output"""
    output_notebook()


def visualize_patches(fixed_tile, moving_tile, transformed_tile):
    """
    Visualize fixed, moving, and transformed image patches
    
    Args:
        fixed_tile: Fixed image patch
        moving_tile: Moving image patch  
        transformed_tile: Transformed image patch
    """
    _, axs = plt.subplots(1, 3, figsize=(15, 10))
    
    axs[0].imshow(fixed_tile, cmap="gray")
    axs[0].set_title("Fixed Tile")
    axs[0].axis('off')
    
    axs[1].imshow(moving_tile, cmap="gray")
    axs[1].set_title("Moving Tile")
    axs[1].axis('off')
    
    axs[2].imshow(transformed_tile, cmap="gray")
    axs[2].set_title("Rigid Transformed Tile")
    axs[2].axis('off')
    
    plt.show()


def create_nuclei_overlay_plot(moving_df, fixed_df, title="Fixed vs Moving Nuclei Coordinates"):
    """
    Create interactive Bokeh plot for nuclei coordinates overlay
    
    Args:
        moving_df: DataFrame with moving nuclei coordinates
        fixed_df: DataFrame with fixed nuclei coordinates
        title: Plot title
        
    Returns:
        Bokeh figure object
    """
    # Ensure 'area' column exists
    if 'area' not in moving_df.columns:
        moving_df['area'] = 1.0
    if 'area' not in fixed_df.columns:
        fixed_df['area'] = 1.0
    
    # Create ColumnDataSources
    source_moving = ColumnDataSource(moving_df)
    source_fixed = ColumnDataSource(fixed_df)
    
    # Create figure
    p = figure(
        title=title,
        x_axis_label='Global X', 
        y_axis_label='Global Y',
        width=VisualizationParams.FIGURE_WIDTH,
        height=VisualizationParams.FIGURE_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom"
    )
    
    # Plot fixed nuclei
    p.triangle('global_x', 'global_y',
               source=source_fixed,
               size=VisualizationParams.POINT_SIZE_MEDIUM,
               fill_color=VisualizationParams.FIXED_COLOR,
               fill_alpha=VisualizationParams.ALPHA,
               line_color=None,
               legend_label='Fixed')
    
    # Plot moving nuclei
    p.circle('global_x', 'global_y',
             source=source_moving,
             size=VisualizationParams.POINT_SIZE_SMALL,
             fill_color=VisualizationParams.MOVING_COLOR,
             fill_alpha=VisualizationParams.ALPHA,
             line_color=None,
             legend_label='Moving')
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Global X", "@global_x{0.00}"),
        ("Global Y", "@global_y{0.00}"),
        ("Area", "@area"),
    ])
    p.add_tools(hover)
    
    # Flip Y-axis to match image coordinates
    p.y_range.flipped = True
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p


def create_registration_comparison_plot(fixed_df, moving_df, moving_rigid_df, 
                                      moving_nonrigid_df=None):
    """
    Create comparison plot showing original, rigid, and non-rigid registration
    
    Args:
        fixed_df: Fixed nuclei DataFrame
        moving_df: Original moving nuclei DataFrame
        moving_rigid_df: Rigid registered moving nuclei DataFrame
        moving_nonrigid_df: Non-rigid registered moving nuclei DataFrame (optional)
        
    Returns:
        Bokeh figure object
    """
    # Create ColumnDataSources
    source_fixed = ColumnDataSource(fixed_df)
    source_moving = ColumnDataSource(moving_df)
    source_moving_rigid = ColumnDataSource(moving_rigid_df)
    
    # Create figure
    p = figure(
        title="Registration Comparison: Original vs Rigid vs Non-Rigid",
        x_axis_label='Global X',
        y_axis_label='Global Y',
        width=VisualizationParams.FIGURE_WIDTH,
        height=VisualizationParams.FIGURE_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom"
    )
    
    # Plot fixed nuclei
    p.triangle('global_x', 'global_y',
               source=source_fixed,
               size=VisualizationParams.POINT_SIZE_MEDIUM,
               fill_color=VisualizationParams.FIXED_COLOR,
               fill_alpha=VisualizationParams.ALPHA,
               line_color=None,
               legend_label='Fixed')
    
    # Plot original moving nuclei
    p.circle('global_x', 'global_y',
             source=source_moving,
             size=VisualizationParams.POINT_SIZE_SMALL,
             fill_color=VisualizationParams.MOVING_COLOR,
             fill_alpha=VisualizationParams.ALPHA,
             line_color=None,
             legend_label='Moving (Original)')
    
    # Plot rigid registered moving nuclei
    p.square('global_x', 'global_y',
             source=source_moving_rigid,
             size=VisualizationParams.POINT_SIZE_MEDIUM,
             fill_color=VisualizationParams.RIGID_COLOR,
             fill_alpha=0.5,
             line_color=None,
             legend_label='Moving (Rigid Registered)')
    
    # Plot non-rigid registered nuclei if provided
    if moving_nonrigid_df is not None:
        source_moving_nonrigid = ColumnDataSource(moving_nonrigid_df)
        p.diamond('global_x', 'global_y',
                  source=source_moving_nonrigid,
                  size=VisualizationParams.POINT_SIZE_LARGE,
                  fill_color=VisualizationParams.NONRIGID_COLOR,
                  fill_alpha=0.5,
                  line_color=None,
                  legend_label='Moving (Non-Rigid Registered)')
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Global X", "@global_x{0.00}"),
        ("Global Y", "@global_y{0.00}"),
        ("Area", "@area"),
    ])
    p.add_tools(hover)
    
    # Flip Y-axis and configure appearance
    p.y_range.flipped = True
    p.xgrid.visible = False
    p.ygrid.visible = False
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p


def visualize_shape_aware_registration(registrator_obj, title="Shape-Aware Point Set Registration"):
    """
    Visualize shape-aware registration results using the built-in method
    
    Args:
        registrator_obj: ShapeAwarePointSetRegistration object after registration
        title: Plot title
        
    Returns:
        Bokeh figure object
    """
    if registrator_obj.registered_points is None:
        raise ValueError("Registration must be performed before visualization")
    
    # Create figure
    p = figure(
        title=title,
        x_axis_label='Global X',
        y_axis_label='Global Y',
        width=VisualizationParams.FIGURE_WIDTH,
        height=VisualizationParams.FIGURE_HEIGHT,
        tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        active_scroll="wheel_zoom"
    )
    
    # Prepare data sources
    fixed_source = ColumnDataSource(registrator_obj.fixed_points)
    moving_orig_source = ColumnDataSource(registrator_obj.moving_points)
    moving_reg_source = ColumnDataSource(registrator_obj.registered_points)
    
    # Plot fixed points
    p.triangle(
        'global_x', 'global_y',
        source=fixed_source,
        size=3.5,
        fill_color='blue',
        fill_alpha=0.7,
        line_color=None,
        legend_label='Fixed'
    )
    
    # Plot original moving points
    p.circle(
        'global_x', 'global_y',
        source=moving_orig_source,
        size=2.5,
        fill_color='red',
        fill_alpha=0.3,
        line_color=None,
        legend_label='Moving (Original)'
    )
    
    # Plot registered moving points
    reg_circles = p.circle(
        'registered_x', 'registered_y',
        source=moving_reg_source,
        size=3,
        fill_color='green',
        fill_alpha=0.7,
        line_color='black',
        line_alpha=0.5,
        line_width=0.5,
        legend_label='Moving (Registered)'
    )
    
    # Draw correspondence lines
    if registrator_obj.correspondence_indices is not None:
        step = max(1, len(registrator_obj.registered_points) // 100)
        for i in range(0, len(registrator_obj.registered_points), step):
            x0 = registrator_obj.registered_points.iloc[i]['registered_x']
            y0 = registrator_obj.registered_points.iloc[i]['registered_y']
            idx = registrator_obj.correspondence_indices[i]
            x1 = registrator_obj.fixed_points.iloc[idx]['global_x']
            y1 = registrator_obj.fixed_points.iloc[idx]['global_y']
            p.line([x0, x1], [y0, y1], line_color='black', line_alpha=0.2, line_width=0.5)
    
    # Add hover tool
    hover = HoverTool(
        renderers=[reg_circles],
        tooltips=[
            ("Original X", "@global_x{0.00}"),
            ("Original Y", "@global_y{0.00}"),
            ("Registered X", "@registered_x{0.00}"),
            ("Registered Y", "@registered_y{0.00}"),
            ("Area", "@area"),
            ("Set", "Moving (Registered)")
        ]
    )
    p.add_tools(hover)
    
    # Flip Y-axis and configure
    p.y_range.flipped = True
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p


def create_method_comparison_plot(fixed_df, moving_df, icp_registered_df, shape_aware_registered_df):
    """
    Create comparison plot between ICP and Shape-Aware registration methods
    
    Args:
        fixed_df: Fixed nuclei DataFrame
        moving_df: Original moving nuclei DataFrame
        icp_registered_df: ICP registered moving nuclei DataFrame
        shape_aware_registered_df: Shape-aware registered moving nuclei DataFrame
        
    Returns:
        Bokeh figure object
    """
    # Create ColumnDataSources
    source_fixed = ColumnDataSource(fixed_df)
    source_moving = ColumnDataSource(moving_df)
    source_icp = ColumnDataSource(icp_registered_df)
    source_shape_aware = ColumnDataSource(shape_aware_registered_df)
    
    # Create figure
    p = figure(
        title="Registration Method Comparison: ICP vs Shape-Aware",
        x_axis_label='Global X',
        y_axis_label='Global Y',
        width=VisualizationParams.FIGURE_WIDTH,
        height=VisualizationParams.FIGURE_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom"
    )
    
    # Plot fixed nuclei
    p.triangle('global_x', 'global_y',
               source=source_fixed,
               size=VisualizationParams.POINT_SIZE_MEDIUM,
               fill_color='blue',
               fill_alpha=0.7,
               line_color=None,
               legend_label='Fixed')
    
    # Plot original moving nuclei
    p.circle('global_x', 'global_y',
             source=source_moving,
             size=VisualizationParams.POINT_SIZE_SMALL,
             fill_color='red',
             fill_alpha=0.3,
             line_color=None,
             legend_label='Moving (Original)')
    
    # Plot ICP registered nuclei
    p.square('global_x', 'global_y',
             source=source_icp,
             size=VisualizationParams.POINT_SIZE_MEDIUM,
             fill_color='green',
             fill_alpha=0.6,
             line_color=None,
             legend_label='Moving (ICP)')
    
    # Plot shape-aware registered nuclei
    p.diamond('registered_x', 'registered_y',
              source=source_shape_aware,
              size=VisualizationParams.POINT_SIZE_MEDIUM,
              fill_color='orange',
              fill_alpha=0.6,
              line_color=None,
              legend_label='Moving (Shape-Aware)')
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Global X", "@global_x{0.00}"),
        ("Global Y", "@global_y{0.00}"),
        ("Area", "@area"),
    ])
    p.add_tools(hover)
    
    # Flip Y-axis and configure appearance
    p.y_range.flipped = True
    p.xgrid.visible = False
    p.ygrid.visible = False
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p


def visualize_transformed_image(transformed_image):
    """
    Display transformed image using matplotlib
    
    Args:
        transformed_image: Transformed image array
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(transformed_image)
    plt.title("Transformed Image")
    plt.axis('off')
    plt.show()


def create_detailed_nuclei_plot_with_colormaps(moving_df, fixed_df):
    """
    Create detailed nuclei plot with color mapping by area
    
    Args:
        moving_df: Moving nuclei DataFrame
        fixed_df: Fixed nuclei DataFrame
        
    Returns:
        Bokeh figure object
    """
    # Create ColumnDataSources
    source_moving = ColumnDataSource(moving_df)
    source_fixed = ColumnDataSource(fixed_df)
    
    # Color mappers
    color_mapper_moving = linear_cmap('area', Viridis256, 
                                    low=moving_df['area'].min(), 
                                    high=moving_df['area'].max())
    color_mapper_fixed = linear_cmap('area', Inferno256, 
                                   low=fixed_df['area'].min(), 
                                   high=fixed_df['area'].max())
    
    # Create figure
    p = figure(
        title="Fixed vs Moving Nuclei Coordinates (Color by Area)",
        x_axis_label='Global X',
        y_axis_label='Global Y',
        width=VisualizationParams.FIGURE_WIDTH,
        height=650,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom"
    )
    
    # Plot moving nuclei with color mapping
    moving_renderer = p.circle('global_x', 'global_y',
                              source=source_moving,
                              size=VisualizationParams.POINT_SIZE_SMALL,
                              fill_color=color_mapper_moving,
                              fill_alpha=VisualizationParams.ALPHA,
                              line_color=None,
                              legend_label='Moving')
    
    # Plot fixed nuclei with color mapping
    fixed_renderer = p.triangle('global_x', 'global_y',
                               source=source_fixed,
                               size=2.5,
                               fill_color=color_mapper_fixed,
                               fill_alpha=VisualizationParams.ALPHA,
                               line_color=None,
                               legend_label='Fixed')
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Global X", "@global_x{0.00}"),
        ("Global Y", "@global_y{0.00}"),
        ("Area", "@area"),
    ], renderers=[moving_renderer, fixed_renderer])
    p.add_tools(hover)
    
    # Flip Y-axis
    p.y_range.flipped = True
    
    # Add color bars
    color_bar_moving = ColorBar(color_mapper=color_mapper_moving['transform'], 
                               title="Area (Moving)")
    color_bar_fixed = ColorBar(color_mapper=color_mapper_fixed['transform'], 
                              title="Area (Fixed)")
    p.add_layout(color_bar_moving, 'right')
    p.add_layout(color_bar_fixed, 'below')
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return p


def show_plot(plot):
    """
    Display a Bokeh plot
    
    Args:
        plot: Bokeh figure object
    """
    show(plot)