"""
Imports for CORE registration
"""

# Core libraries
import cv2
import numpy as np
import pandas as pd
import os
import sys
import datetime
import builtins
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import scipy.ndimage
from scipy.interpolate import Rbf, RBFInterpolator, griddata
from scipy.ndimage import gaussian_filter, map_coordinates

# Image processing and registration
from skimage.registration import phase_cross_correlation
from skimage import transform, color
from PIL import Image
import SimpleITK as sitk

# Machine learning and point cloud processing
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from pycpd import DeformableRegistration

# TIA Toolbox
import tiatoolbox
from tiatoolbox.wsicore.wsireader import WSIReader, TransformedWSIReader
from tiatoolbox.tools import patchextraction
from tiatoolbox.tools.registration.wsi_registration import AffineWSITransformer

# Bokeh for interactive visualization
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, Title, Legend
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256, Inferno256
from bokeh.io import output_notebook

# Custom modules (assuming they exist in src/)
import sys
import os
import core.preprocessing.preprocessing
import core.evaluation.evaluation
import core.registration.rigid
import core.registration.nonrigid
import core.utils.util
import core.config
import core.preprocessing.padding
