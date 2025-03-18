from typing import Dict
from jaxtyping import Float

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange

def sph2cart(
    azi: Float[Tensor, 'B'],
    ele: Float[Tensor, 'B'],
    r: Float[Tensor, 'B']
) -> Float[Tensor, 'B 3']:
    # z-up, y-right, x-back
    rcos = r * torch.cos(ele)
    pos_cart = torch.stack([
        rcos * torch.cos(azi),
        rcos * torch.sin(azi),
        r * torch.sin(ele)
    ], dim=1)

    return pos_cart

def get_viewport_matrix(
    width: int,
    height: int,
    batch_size: int=1,
    device: torch.device=None,
) -> Float[Tensor, 'B 4 4']:
    N = torch.tensor(
        [[width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1/2, 1/2],
        [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device
    )[None].repeat(batch_size, 1, 1)
    return N

def get_projection_matrix(
    fovy: Float[Tensor, 'B'],
    aspect_wh: float,
    near: float,
    far: float
) -> Float[Tensor, 'B 4 4']:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def camera_lookat(
    eye: Float[Tensor, 'B 3'],
    target: Float[Tensor, 'B 3'],
    up: Float[Tensor, 'B 3']
) -> Float[Tensor, 'B 4 4']:
    B = eye.shape[0]
    f = F.normalize(eye - target)
    l = F.normalize(torch.linalg.cross(up, f))
    u = F.normalize(torch.linalg.cross(f, l))

    R = torch.stack((l, u, f), dim=1)  # B 3 3
    M_R = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_R[..., :3, :3] = R

    T = - eye
    M_T = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_T[..., :3, 3] = T

    return (M_R @ M_T).to(dtype=torch.float32)

def focal_length_to_fov(
    focal_length: float,
    censor_length: float = 24.
) -> float:
    return 2 * np.arctan(censor_length / focal_length / 2.)

def forward_warper(
    image: Float[Tensor, 'B C H W'],
    screen: Float[Tensor, 'B (H W) 2'],
    pcd: Float[Tensor, 'B (H W) 4'],
    mvp_mtx: Float[Tensor, 'B 4 4'],
    viewport_mtx: Float[Tensor, 'B 4 4'],
    alpha: float = 0.5
) -> Dict[str, Tensor]:
    H, W = image.shape[2:4]

    # Projection.
    points_c = pcd @ mvp_mtx.mT
    points_ndc = points_c / points_c[..., 3:4]
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e+4

    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)

    # Splatting.
    warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.).all(dim=1, keepdim=True).to(image.dtype)
    flow2 = rearrange(coords_new[..., :2], 'b (h w) c -> b c h w', h=H, w=W)

    output = dict(
        warped=warped,
        mask=mask,
        correspondence=flow2
    )

    return output


def convert_left_to_right(left_embed, disparity, left_image, random_ratio=None):
    # Get the height, width, and channels from the left embedding
    _, height, width = left_embed.shape

    # Initialize tensors for right_embed, converted_right_image, and mask
    # right_embed = torch.full_like(left_embed, 255)
    # converted_right_image = torch.full_like(left_image, 255)
    right_embed = torch.ones_like(left_embed)
    converted_right_image = torch.ones_like(left_image)    
    mask = torch.ones((height, width), dtype=torch.uint8, device=left_embed.device)

    # Round the disparity and convert to int
    disparity_rounded = torch.round(disparity).squeeze(0).long()  # [h, w]

    # Loop through the image dimensions and apply the conversion
    for y in range(height):
        for x in range(width):
            new_x = x - disparity_rounded[y, x]

            if 0 <= new_x < width:# and disparity_rounded[y, x] > 0:
                right_embed[:, y, new_x] = left_embed[:, y, x]
                converted_right_image[:, y, new_x] = left_image[:, y, x]
                mask[y, new_x] = 0  # Mark as valid in the mask
    # print(f"Mask sum before: {mask.sum()}")
    # Apply random mask if drop_ratio is set
    if random_ratio is not None:
        print(f"Random ratio: {random_ratio}")
        # Create a random mask with values ranging from 0 (invalid) to 1 (valid)
        random_mask = torch.bernoulli(torch.full((height, width), 1 - random_ratio, device=left_embed.device)).byte()
        # Perform a logical AND operation with the mask from the function
        mask = mask | random_mask

        # Apply the final mask to right_embed, converted_right_image, and disparity
        right_embed[:, mask == 1] = 255  # Set masked out locations to 255 in the right embed
        converted_right_image[:, mask == 1] = 255  # Set masked out locations to 255 in the converted right image
        disparity[:, mask == 1] = 0  # Set masked out locations in the original disparity to 0
        # print(f"Mask sum after: {mask.sum()}")
    return right_embed, mask, converted_right_image, disparity


def convert_left_to_right_torch(left_embed, disparity, left_image, random_ratio=None):
    """
    Convert left features to right features based on disparity values.
    
    Args:
        left_embed (torch.Tensor): [c, h, w] tensor representing left feature embeddings.
        disparity (torch.Tensor): [1, h, w] tensor of disparity values.
        left_image (torch.Tensor): [3, h, w] tensor representing the left image.

    Returns:
        right_embed (torch.Tensor): [c, h, w] tensor for the right feature embeddings.
        mask (torch.Tensor): [h, w] binary mask (1 = invalid, 0 = valid).
        converted_right_image (torch.Tensor): [3, h, w] tensor for the right image.
        disparity (torch.Tensor): [1, h, w] tensor for the disparity.
    """
    # Get the height, width, and channels from the left embedding
    _, height, width = left_embed.shape

    # Initialize tensors for right_embed, converted_right_image, and mask
    right_embed = torch.zeros_like(left_embed)
    # converted_right_image = torch.zeros_like(left_image)
    converted_right_image = -torch.ones_like(left_image)
    mask = torch.ones((height, width), dtype=torch.uint8, device=left_embed.device)

    # Round the disparity and convert to int
    disparity_rounded = torch.round(disparity).squeeze(0).long()  # [h, w]

    # Iterate over width and process each column for all rows
    for x in range(width):
        new_x = x - disparity_rounded[:, x]

        valid_indices = (new_x >= 0) & (new_x < width) #& (disparity_rounded[:, x] > 0)
        valid_new_x = new_x[valid_indices]
        valid_y = torch.arange(height, device=left_embed.device)[valid_indices]

        right_embed[:, valid_y, valid_new_x] = left_embed[:, valid_y, x]
        converted_right_image[:, valid_y, valid_new_x] = left_image[:, valid_y, x]
        mask[valid_y, valid_new_x] = 0  # Mark as valid in the mask

    return right_embed, mask, converted_right_image, disparity