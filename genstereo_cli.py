import os
import argparse
import glob
import time
import warnings
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
import cv2
import traceback
from tqdm import tqdm
from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2

from genstereo import (
    GenStereo, 
    AdaptiveFusionLayer,
    convert_left_to_right_torch  # Now available via package
)
from genstereo.ops import convert_left_to_right_torch
from genstereo.tiling_utils import pad_to_tile, split_into_tiles, merge_tiles

warnings.filterwarnings("ignore")

IMAGE_SIZE = 512  # Base resolution for depth estimation

class GenStereoPipeline:
    def __init__(self, device: str = "cuda", tile_size: int = 512):
        self.device = device
        self.tile_size = tile_size
        self.genstereo = self._load_genstereo()
        self.fusion_model = self._load_fusion_model()
        self.dam2 = self._load_dam2()
        
    def _load_genstereo(self):
        genstereo_cfg = dict(
            pretrained_model_path='checkpoints',
            checkpoint_name='genstereo',
            half_precision_weights=True
        )
        return GenStereo(cfg=genstereo_cfg, device=self.device)

    def _load_fusion_model(self):
        fusion_model = AdaptiveFusionLayer()
        fusion_checkpoint = os.path.join('checkpoints', 'genstereo', 'fusion_layer.pth')
        fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location='cpu'))
        return fusion_model.to(self.device).eval()

    def _load_dam2(self):
        model_configs = {
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            }
        }
        
        dam2 = DepthAnythingV2(**model_configs['vitl'])
        checkpoint_path = 'checkpoints/depth_anything_v2_vitl.pth'
        dam2.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        return dam2.to(self.device).eval()

    def _estimate_depth(self, image: Image) -> np.ndarray:
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image_bgr.shape[:2]
        
        # Maintain aspect ratio while resizing
        scale = IMAGE_SIZE / min(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h))
        
        # Initial depth estimation
        with torch.no_grad():
            depth_map = self.dam2.infer_image(resized)
        
        # Resize back to original dimensions
        depth_map = cv2.resize(depth_map, (orig_w, orig_h))

        # Guided filtering with positional arguments
        guidance = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
        depth_map = cv2.ximgproc.guidedFilter(
            guidance,  # Guide image (positional)
            depth_map.astype(np.float32),  # Source image (positional)
            16,  # radius
            100,  # eps
            -1  # dDepth
        )

        return depth_map

    def _calculate_disparity(self, depth: torch.Tensor, scale_factor: float, 
                            global_min: float = None, global_max: float = None) -> torch.Tensor:
        if global_min is None or global_max is None:
            global_min = depth.min()
            global_max = depth.max()
            
        norm_disp = (depth - global_min) / (global_max - global_min + 1e-9)
        return norm_disp * scale_factor / 100 * self.tile_size

    def process_image(self, image_path: str, output_dir: str, scale_factor: float = 15.0):
        torch.cuda.empty_cache()
        try:
            # Load and prepare image
            orig_image = Image.open(image_path).convert('RGB')
            orig_height, orig_width = orig_image.height, orig_image.width
            
            # Estimate depth for full image
            depth_map = self._estimate_depth(orig_image)
            
            # Validate depth map dimensions
            if len(depth_map.shape) != 2:
                print(f"Skipping {image_path}: Invalid depth map dimensions {depth_map.shape}")
                return
                
            # Calculate global disparity stats from original depth map
            global_disp_min = depth_map.min().item()
            global_disp_max = depth_map.max().item()
            
            # Pad and split both image and depth with overlap
            img_padded, img_padding = pad_to_tile(np.array(orig_image), self.tile_size + 64)
            depth_padded, depth_padding = pad_to_tile(depth_map, self.tile_size + 64)
            
            if img_padding != depth_padding:
                raise ValueError(f"Padding mismatch: Image {img_padding} vs Depth {depth_padding}")
            
            img_tiles = split_into_tiles(img_padded, self.tile_size)
            depth_tiles = split_into_tiles(depth_padded, self.tile_size, overlap=64)
            
            # Validate tile counts match
            if len(img_tiles) != len(depth_tiles):
                raise ValueError(f"Tile count mismatch: {len(img_tiles)} image vs {len(depth_tiles)} depth tiles")
            
            processed_tiles = []
            warped_tiles = []
            
            print(f"Total RGB tiles: {len(img_tiles)}, Depth tiles: {len(depth_tiles)}")
            # Process aligned tile pairs with global normalization
            for (img_tile, img_pos), (depth_tile, depth_pos) in zip(tqdm(img_tiles, desc="Processing tiles"), depth_tiles):
                try:
                    print(f"Processing tile - Image @ {img_pos}, Depth @ {depth_pos}")
                    if img_pos != depth_pos:
                        print(f"Position mismatch: {img_pos} vs {depth_pos} - Using nearest alignment")
                        continue  # Skip instead of raising error
                    
                    # Convert to tensors
                    img_tensor = torch.from_numpy(img_tile).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)
                    
                    depth_tensor = torch.from_numpy(depth_tile).float()
                    depth_tensor = depth_tensor.unsqueeze(0)  # Add batch dim
                    if depth_tensor.ndim == 3:
                        depth_tensor = depth_tensor.unsqueeze(1)  # Add channel dim
                    
                    # Calculate disparity with GLOBAL normalization
                    with torch.no_grad():
                        disp = self._calculate_disparity(
                            depth_tensor, 
                            scale_factor,
                            global_min=global_disp_min,
                            global_max=global_disp_max
                        )
                        
                    # Rest of processing remains the same
                    renders = self.genstereo(
                        src_image=Image.fromarray(img_tile),
                        src_disparity=disp.to(self.device),
                        ratio=None
                    )
                    
                    # Fusion process
                    warped = (renders['warped'] + 1) / 2
                    synthesized = renders['synthesized']
                    mask = renders['mask']
                    
                    fusion_image = self.fusion_model(
                        synthesized.float(), 
                        warped.float(), 
                        mask.float()
                    )

                    # For processed_tiles
                    processed_tiles.append((
                        fusion_image[0].detach().cpu().numpy().transpose(1, 2, 0),  # Added .detach()
                        img_pos
                    ))

                    # For warped_tiles
                    warped_tiles.append((
                        warped[0].detach().cpu().numpy().transpose(1, 2, 0),  # Added .detach()
                        img_pos
                    ))
                    
                except Exception as tile_error:
                    print(f"\nTile processing failed at position {img_pos}:")
                    traceback.print_exc()
                    continue

            # Save results (existing code remains)
            if processed_tiles:
                self._save_output(processed_tiles, (orig_height, orig_width), img_padding, output_dir, image_path, "output")
                self._save_output(warped_tiles, (orig_height, orig_width), img_padding, output_dir, image_path, "warped")
            else:
                print(f"No valid tiles processed for {image_path}")
                
        except Exception as e:
            print(f"\nCritical error processing {image_path}:")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    def postprocess(self, merged: np.ndarray) -> np.ndarray:
        """Edge-aware smoothing for final output"""
        try:
            # Convert to 8-bit with proper scaling
            merged_8bit = (merged * 255).astype(np.uint8)
            
            # Apply guided filter
            filtered_8bit = cv2.ximgproc.guidedFilter(
                guide=merged_8bit,
                src=merged_8bit,
                radius=16,
                eps=100
            )
            
            # Convert back to [0,1] range
            return filtered_8bit.astype(np.float32) / 255.0
            
        except AttributeError:
            # Fallback without OpenCV's guided filter
            return cv2.GaussianBlur(merged, (21, 21), 0)

    def _save_output(self, tiles_with_pos, orig_size, padding, output_dir, image_path, suffix):
        if not tiles_with_pos:
            print(f"No tiles to save for {suffix}")
            return
            
        try:
            # Merge tiles
            merged = merge_tiles(tiles_with_pos, orig_size, padding, self.tile_size, 64)
            
            # Construct output path
            base_name = os.path.basename(image_path)
            output_name = f"{os.path.splitext(base_name)[0]}_{suffix}.png"
            output_path = os.path.join(output_dir, output_name)
            
            # Post-processing
            merged = self.postprocess(merged)
            
            # Convert and validate final output
            merged_8bit = (merged * 255).astype(np.uint8)
            
            if merged_8bit.max() == 0:
                raise ValueError(f"Final {suffix} output is all zeros")
                
            print(f"{suffix.capitalize()} range: {merged_8bit.min()} - {merged_8bit.max()}")
            
            # Save as BGR
            cv2.imwrite(output_path, cv2.cvtColor(merged_8bit, cv2.COLOR_RGB2BGR))
            
        except Exception as save_error:
            print(f"Save failed for {suffix}: {str(save_error)}")


def process_batch(input_path: str, output_dir: str, scale_factor: float = 15.0, tile_size: int = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    pipeline = GenStereoPipeline(device=device, tile_size=tile_size)
    
    if os.path.isdir(input_path):
        image_paths = glob.glob(os.path.join(input_path, "*.jpg")) + glob.glob(os.path.join(input_path, "*.png"))
    else:
        image_paths = [input_path]

    print(f"Found {len(image_paths)} images to process")
    
    for path in tqdm(image_paths, desc="Total progress"):
        pipeline.process_image(path, output_dir, scale_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenStereo Batch Processor")
    parser.add_argument("input", type=str, help="Input file or directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--scale", type=float, default=15.0, 
                      help="Disparity scale factor (15.0 by default)")
    parser.add_argument("--tile_size", type=int, default=512,
                      help="Processing tile size (512 by default)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path {args.input} does not exist")
    
    start_time = time.time()
    process_batch(args.input, args.output_dir, args.scale, args.tile_size)
    print(f"\nProcessing completed in {time.time()-start_time:.2f} seconds")