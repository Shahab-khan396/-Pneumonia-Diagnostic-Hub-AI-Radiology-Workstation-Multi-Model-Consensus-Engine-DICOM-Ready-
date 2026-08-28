import cv2
import numpy as np
from pathlib import Path
from typing import Union
from config import IMG_SIZE


def preprocess_image(
    image_source: Union[str, Path, bytes, np.ndarray], 
    img_size: int = IMG_SIZE
) -> np.ndarray:
    """
    Preprocess an image for Pneumonia CNN model inference.
    
    Pipeline:
      1. Load as grayscale (matches training notebook data pipeline).
      2. Resize to (img_size, img_size).
      3. Stack grayscale into 3-channel pseudo-RGB.
      4. Normalize pixel values to [0.0, 1.0].
      5. Reshape to batch format (1, img_size, img_size, 3).
      
    Args:
        image_source: File path (str/Path), raw bytes, or existing numpy array.
        img_size: Target square image dimension (default 128).
        
    Returns:
        np.ndarray: Batch tensor of shape (1, img_size, img_size, 3) in float32.
        
    Raises:
        ValueError: If the image cannot be read or decoded.
    """
    img_gray = None
    
    if isinstance(image_source, (str, Path)):
        file_path_str = str(image_source)
        img_gray = cv2.imread(file_path_str, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_source, bytes):
        nparr = np.frombuffer(image_source, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 3 and image_source.shape[2] == 3:
            img_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
        elif image_source.ndim == 2:
            img_gray = image_source
        else:
            raise ValueError(f"Unsupported array shape for image: {image_source.shape}")
    else:
        raise ValueError(f"Unsupported image_source type: {type(image_source)}")
        
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Failed to decode image. File may be corrupted or not a valid image format.")
        
    # Resize to model input dimensions
    resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Merge into 3 channels for transfer learning backbones
    img_rgb = cv2.merge([resized, resized, resized])
    
    # Normalize to [0.0, 1.0]
    normalized = img_rgb.astype(np.float32) / 255.0
    
    # Reshape to (1, img_size, img_size, 3)
    batch_tensor = np.expand_dims(normalized, axis=0)
    
    return batch_tensor
