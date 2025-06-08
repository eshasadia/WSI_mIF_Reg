from typing import Iterable, Tuple, List, Dict
import math
import numpy as np

def calculate_pad_value(size_1: Iterable[int], size_2: Iterable[int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Calculates the padding required to make two images the same size."""
    y_size_1, x_size_1 = size_1
    y_size_2, x_size_2 = size_2
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
        
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
        
    return pad_1, pad_2

def apply_padding_landmarks(landmarks: np.ndarray, pad: List[Tuple[int, int]]) -> np.ndarray:
    """Applies padding to landmark coordinates."""
    padded_landmarks = landmarks.copy()
    padded_landmarks[:, 0] += pad[1][0]  # x coordinate
    padded_landmarks[:, 1] += pad[0][0]  # y coordinate
    return padded_landmarks

def pad_images(
    image_1: np.ndarray,
    image_2: np.ndarray,
    pad_value: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """Pads two images to the same size and optionally adjusts landmarks accordingly."""
    y_size_1, x_size_1 = image_1.shape[:2]
    y_size_2, x_size_2 = image_2.shape[:2]
    
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    
    image_1 = np.pad(image_1, ((pad_1[0][0], pad_1[0][1]), (pad_1[1][0], pad_1[1][1]), (0, 0)), 
                     mode='constant', constant_values=pad_value)
    image_2 = np.pad(image_2, ((pad_2[0][0], pad_2[0][1]), (pad_2[1][0], pad_2[1][1]), (0, 0)), 
                     mode='constant', constant_values=pad_value)
    
  
    padding_params = {
        'pad_1': pad_1,
        'pad_2': pad_2
    }
    
    return image_1, image_2, padding_params

def pad_landmarks(
    padding_params,
    landmarks_1: np.ndarray = None,
    landmarks_2: np.ndarray = None,
    
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
   
    
    
    
    landmarks_1_padded = apply_padding_landmarks(landmarks_1, padding_params['pad_1']) if landmarks_1 is not None else None
    landmarks_2_padded = apply_padding_landmarks(landmarks_2, padding_params['pad_2']) if landmarks_2 is not None else None
    
   
    return  landmarks_1_padded, landmarks_2_padded