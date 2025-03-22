import torch
import cv2
import numpy as np
from torch.nn.functional import pad
from typing import Tuple, List

def pad_to_tile(image: np.ndarray, tile_size: int = 512) -> Tuple[np.ndarray, Tuple]:
    h, w = image.shape[:2]
    
    # Ensure minimum padding
    pad_h = max(tile_size - h, (tile_size - (h % tile_size)) % tile_size)
    pad_w = max(tile_size - w, (tile_size - (w % tile_size)) % tile_size)
    
    padding = ((0, pad_h), (0, pad_w), (0, 0)) if len(image.shape) == 3 else ((0, pad_h), (0, pad_w))
    pad_mode = 'reflect' if len(image.shape) == 3 else 'edge'
    padded = np.pad(image, padding, mode=pad_mode)
    
    return padded, (pad_h, pad_w)

def calculate_content_aware_overlap(image: np.ndarray, tile_size: int) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0
    
    # Dynamic overlap range 32-128 pixels based on edge density
    base_overlap = 64
    dynamic_overlap = int(base_overlap * (1 + edge_density))
    return np.clip(dynamic_overlap, 32, 128)

def split_into_tiles(padded: np.ndarray, tile_size: int = 512, overlap: int = None) -> List[Tuple[np.ndarray, tuple]]:
    """Handle edge cases for small images"""
    h, w = padded.shape[:2]
    
    # Ensure minimum dimensions
    h = max(h, tile_size)
    w = max(w, tile_size)
    
    # Calculate safe overlap
    if overlap is None or overlap >= tile_size:
        overlap = min(64, tile_size-1)  # Hard cap at 64px
    
    stride = max(1, tile_size - overlap)  # Prevent zero/negative stride
    
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract tile even if smaller than tile_size
            tile = padded[y:y_end, x:x_end]
            
            # Only pad if needed
            pad_y = max(0, tile_size - (y_end - y))
            pad_x = max(0, tile_size - (x_end - x))
            
            if pad_y > 0 or pad_x > 0:
                pad_mode = 'reflect' if len(padded.shape) == 3 else 'edge'
                tile = np.pad(tile,
                            ((0, pad_y), (0, pad_x)) if len(padded.shape) == 2 
                            else ((0, pad_y), (0, pad_x), (0, 0)),
                            mode=pad_mode)
            
            tiles.append((tile, (y, x)))
    return tiles

def create_blend_mask(tile_size: int, overlap: int=64) -> np.ndarray:
    mask = np.ones((tile_size, tile_size))
    
    # Horizontal featheredges
    mask[:, :overlap] *= np.linspace(0, 1, overlap)
    mask[:, -overlap:] *= np.linspace(1, 0, overlap)
    
    # Vertical featheredges
    mask[:overlap, :] *= np.linspace(0, 1, overlap)[:, None]
    mask[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None]
    
    return mask[..., None]  # Add channel dim for RGB

def merge_tiles(tiles: List[Tuple[np.ndarray, tuple]], 
                original_shape: tuple, 
                padding: tuple, 
                tile_size: int=512,
                overlap: int=64) -> np.ndarray:
    h, w = original_shape
    blend_mask = create_blend_mask(tile_size, overlap)
    
    # Initialize with proper padded dimensions
    merged = np.zeros((h + padding[0], w + padding[1], 3), dtype=np.float32)
    weight_sum = np.zeros((h + padding[0], w + padding[1], 3), dtype=np.float32)
    
    for tile_idx, (tile, (y, x)) in enumerate(tiles):
        # Validate input tile
        if tile.size == 0:
            print(f"Empty tile at position ({y},{x}) - Tile {tile_idx}")
            continue
            
        if np.isnan(tile).any():
            print(f"NaN values in tile at ({y},{x}) - Tile {tile_idx}")
            continue
            
        # Calculate valid region without padding
        valid_h = min(tile_size, h + padding[0] - y)
        valid_w = min(tile_size, w + padding[1] - x)
        
        # Verify valid region dimensions
        if valid_h <= 0 or valid_w <= 0:
            print(f"Invalid region size {valid_h}x{valid_w} at ({y},{x})")
            continue
            
        try:
            # Apply mask to valid region only
            masked_tile = tile[:valid_h, :valid_w] * blend_mask[:valid_h, :valid_w]
            
            # Verify mask application
            if masked_tile.shape != (valid_h, valid_w, 3):
                print(f"Shape mismatch: {masked_tile.shape} vs expected ({valid_h}, {valid_w}, 3)")
                continue
                
            # Accumulate
            merged[y:y+valid_h, x:x+valid_w] += masked_tile
            weight_sum[y:y+valid_h, x:x+valid_w] += blend_mask[:valid_h, :valid_w]
            
        except Exception as e:
            print(f"Error processing tile {tile_idx} at ({y},{x}): {str(e)}")
            continue
    
    # Verify merged output before normalization
    if np.isnan(merged).any():
        print("NaN values detected in merged output before normalization")
        
    if merged.max() < 1e-6:
        print("Warning: Merged output values are near zero")
        
    # Normalize and crop
    merged = np.divide(merged, weight_sum, where=weight_sum > 0)
    merged = np.nan_to_num(merged)  # Handle potential division by zero
    
    # Final validation
    if merged.max() < 0.01:  # 1% of 255 value range
        print("Critical warning: Final output values too low")
        
    return merged[:h, :w]

def seam_aware_merge(merged: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    
    # Create feathering mask
    feather_mask = np.clip(dist_transform / 20, 0, 1)[..., np.newaxis]
    return merged * feather_mask + merged * (1 - feather_mask)
    
def laplacian_pyramid_blend(tiles: List[Tuple[np.ndarray, tuple]], 
                           original_shape: tuple, 
                           padding: tuple,
                           tile_size: int=512,
                           overlap: int=64,
                           levels: int=5) -> np.ndarray:
    
    def build_pyramid(img, levels):
        pyramid = [img.astype(np.float32)]
        for _ in range(levels-1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def collapse_pyramid(pyramid):
        img = pyramid[-1]
        for i in range(len(pyramid)-2, -1, -1):
            img = cv2.pyrUp(img, dstsize=pyramid[i].shape[:2][::-1])
            img += pyramid[i]
        return np.clip(img, 0, 1)

    # Initialize pyramid accumulators
    h, w = original_shape
    blend_mask = create_blend_mask(tile_size, overlap)
    pyramid_accum = [np.zeros((h + padding[0], w + padding[1], 3), dtype=np.float32) 
                    for _ in range(levels)]
    weight_pyramid = [np.zeros((h + padding[0], w + padding[1], 3), dtype=np.float32)
                     for _ in range(levels)]

    for tile, (y, x) in tiles:
        tile_pyramid = build_pyramid(tile, levels)
        mask_pyramid = build_pyramid(blend_mask, levels)
        
        for level in range(levels):
            y_level = y // (2**level)
            x_level = x // (2**level)
            h_level, w_level = tile_pyramid[level].shape[:2]
            
            pyramid_accum[level][y_level:y_level+h_level, x_level:x_level+w_level] += \
                tile_pyramid[level] * mask_pyramid[level]
            weight_pyramid[level][y_level:y_level+h_level, x_level:x_level+w_level] += \
                mask_pyramid[level]

    # Normalize and collapse
    merged = []
    for level in range(levels):
        merged_level = np.divide(pyramid_accum[level], weight_pyramid[level],
                                where=weight_pyramid[level] > 0)
        merged.append(merged_level)
        
    return collapse_pyramid(merged)[:h, :w]
    
def calculate_content_aware_overlap(image: np.ndarray, tile_size: int) -> int:
    """Calculate overlap based on edge density in the image tile"""
    # Handle single-channel (depth) vs RGB images
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.squeeze().astype(np.uint8)
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0  # Normalize to [0,1]
    
    # Dynamic overlap range (32-128 pixels)
    base_overlap = 64
    dynamic_overlap = int(base_overlap * (1 + edge_density))
    return np.clip(dynamic_overlap, 32, 128)