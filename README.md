# CORE - Coarse to Fine Multimodal Registration

A comprehensive workflow for Whole Slide Image (WSI) registration using rigid and non-rigid techniques with nuclei-based analysis.




## Configuration

Edit `config.py` to set your file paths and parameters:

```python
# Update these paths to match your data
SOURCE_WSI_PATH = "/path/to/your/source_wsi.tiff"
TARGET_WSI_PATH = "/path/to/your/target_wsi.tiff"

```

## Usage

### Basic Usage

Run the complete workflow:
```bash
python main.py
```



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

## Output Files

- `comet_nuclei_coordinates_he_7577.csv`: Fixed im
