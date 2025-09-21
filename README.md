# WSI Registration Project

A comprehensive workflow for Whole Slide Image (WSI) registration using rigid and non-rigid techniques with nuclei-based analysis.

## Project Structure

```
wsi_registration/
├── main.py                 # Main execution script
├── config.py              # Configuration and parameters
├── imports.py             # All import statements
├── preprocessing.py       # Image preprocessing functions
├── registration.py        # Registration algorithms
├── evaluation.py          # Evaluation metrics and functions
├── visualization.py       # Bokeh visualization functions
├── nuclei_analysis.py     # Nuclei detection and analysis
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Multi-resolution WSI processing** using TIA Toolbox
- **Rigid registration** using traditional image registration techniques
- **Non-rigid registration** using Coherent Point Drift (CPD)
- **Shape-aware Rigid registration** for point cloud alignment
- **Nuclei detection** in tissue patches
- **Interactive visualization** using Bokeh
- **Comprehensive evaluation** with Target Registration Error (TRE) metrics
- **Displacement field generation** for spatial transformation analysis

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.py` to set your file paths and parameters:

```python
# Update these paths to match your data
SOURCE_WSI_PATH = "/path/to/your/source_wsi.tiff"
TARGET_WSI_PATH = "/path/to/your/target_wsi.tiff"
MOVING_POINTS_PATH = "/path/to/moving_landmarks.csv"
FIXED_POINTS_PATH = "/path/to/fixed_landmarks.csv"
```

## Usage

### Basic Usage

Run the complete workflow:
```bash
python main.py
```

### Custom Usage

You can also import and use individual modules:

```python
from preprocessing import load_wsi_images, preprocess_images
from registration import perform_rigid_registration, perform_icp_registration
from visualization import create_nuclei_overlay_plot

# Load images
source_wsi, target_wsi, source, target = load_wsi_images(source_path, target_path)

# Perform registration
transformed_img, transform = perform_rigid_registration(source, target, source_mask, target_mask)

# Visualize results
plot = create_nuclei_overlay_plot(moving_df, fixed_df)
```

## Workflow Overview

1. **Load WSI Images**: Load source and target whole slide images
2. **Preprocessing**: Extract tissue masks and prepare images
3. **Rigid Registration**: Perform initial coarse alignment
4. **Patch Extraction**: Extract patches for detailed analysis
5. **Nuclei Detection**: Detect nuclei in tissue patches
6. **Shape aware Registration**: Refine alignment using detected nuclei
7. **Non-rigid Registration**: Apply CPD for local deformation
8. **Displacement Field**: Generate smooth transformation field
9. **Evaluation**: Calculate TRE metrics on landmark points
10. **Visualization**: Interactive plots of registration results

## Key Parameters

### Registration Parameters
- `PREPROCESSING_RESOLUTION`: Resolution for initial processing (default: 0.625)
- `REGISTRATION_RESOLUTION`: High-resolution for patch analysis (default: 40)
- `ICP_THRESHOLD`: Maximum correspondence distance for ICP (default: 50000.0)
- `CPD_BETA`: Smoothness parameter for CPD (default: 0.5)
- `CPD_ALPHA`: Regularization parameter for CPD (default: 0.01)

### Nuclei Detection Parameters
- `FIXED_THRESHOLD`: Binary threshold for fixed image nuclei (default: 170)
- `MOVING_THRESHOLD`: Binary threshold for moving image nuclei (default: 180)
- `MIN_NUCLEI_AREA`: Minimum area for nuclei detection (default: 200)
