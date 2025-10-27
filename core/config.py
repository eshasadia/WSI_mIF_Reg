"""
Configuration file for WSI registration project
"""

# File paths
SOURCE_WSI_PATH = "/home/u5552013/Nextcloud/HYRECO/Data/Image/ki67_533.tif"
TARGET_WSI_PATH = "/home/u5552013/Nextcloud/HYRECO/Data/Image/he_533.tif"


# SOURCE_WSI_PATH='/home/u5552013/Nextcloud/ANHIR/COAD_02/scale-25pc/S2.jpg'
# TARGET_WSI_PATH='/home/u5552013/Nextcloud/ANHIR/COAD_02/scale-25pc/S5.jpg'

# Landmark paths
MOVING_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/ki67_533.csv"
FIXED_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/he_533.csv"

# Alternative landmark paths for evaluation
EVAL_MOVING_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/ki67_533.csv"
EVAL_FIXED_POINTS_PATH = "/home/u5552013/Nextcloud/HYRECO/Eval/he_533.csv"

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
# comet
# FIXED_THRESHOLD = 170
# MOVING_THRESHOLD = 180
# hyreco
FIXED_THRESHOLD = 100
MOVING_THRESHOLD = 50
MIN_NUCLEI_AREA = 200
GAMMA_CORRECTION = 0.4

# Registration algorithm parameters
class RegistrationParams:
    # ICP parameters
    ICP_THRESHOLD = 50000.0
    
    # CPD parameters
    CPD_BETA = 0.5  # larger beta increases influence radius (smoother)
    CPD_ALPHA = 0.01  # lower alpha reduces the regularization
    CPD_MAX_ITERATIONS = 200
    CPD_TOLERANCE = 1e-9
    
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