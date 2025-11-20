"""
Configuration file for CORE
"""

# File paths
SOURCE_WSI_PATH='/home/u5552013/Nextcloud/REACTIVAS_vis/1/slides/1.svs'
TARGET_WSI_PATH='/home/u5552013/Nextcloud/REACTIVAS_vis/1/slides/1_ome.tiff'


MOVING_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/ki67_533.csv"
FIXED_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/he_533.csv"

# Output CSV paths
FIXED_NUCLEI_CSV = '/home/u5552013/Nextcloud/HYRECO/Data/nuclei_points/he_533_nuclei.csv'
MOVING_NUCLEI_CSV = '/home/u5552013/Nextcloud/HYRECO/Data/nuclei_points/ki67_533_nuclei.csv'

# Registration parameters
PREPROCESSING_RESOLUTION = 0.625
REGISTRATION_RESOLUTION = 40
PATCH_SIZE = (1000, 1000)
PATCH_STRIDE = (1000, 1000)
VISUALIZATION_SIZE = (5000, 5000)

# Nuclei detection parameters
FIXED_THRESHOLD = 100
MOVING_THRESHOLD = 50
MIN_NUCLEI_AREA = 200
GAMMA_CORRECTION = 0.4

# Registration algorithm parameters
class RegistrationParams:
    # MNN sampling
    MNN_SAMPLE_SIZE = 5000
    
    # Displacement field parameters
    DISPLACEMENT_SIGMA = 10.0
    MAX_DISPLACEMENT = 500.0
    INTERPOLATION_METHOD = 'linear'

# Visualization parameters
class VisualizationParams:
    FIGURE_WIDTH = 900
    FIGURE_HEIGHT = 700
    POINT_SIZE_SMALL = 2
    POINT_SIZE_MEDIUM = 3
    POINT_SIZE_LARGE = 10
    ALPHA = 0.6
    
    # Colors
    FIXED_COLOR = "blue"
    MOVING_COLOR = "red" 
    RIGID_COLOR = "green"
    NONRIGID_COLOR = "orange"
