"""
This file contains utility functions for converting raw satellite images into patches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .filesystem import save_raster_data

def get_patch_indicies(raster: np.ndarray, 
                       patch_size:int, 
                       validator: callable) -> list[tuple[int, int]]:
    """Get indicies of valid patchs in a raster

    Args:
        raster (np.array): A raster stack of shape (bands, height, width)
        patch_size (int): Size of the patch
        validator (function): A function that takes a patch and returns a boolean

    Returns:
        list[tuple[int, int]]: List of tuples like (i, j) where i and j are the indicies of the top left corner of the patch
    """
    patch_indicies = []
    # find the largest height and width that are divisible by patch_size
    max_height = raster.shape[-2] // patch_size * patch_size
    max_width = raster.shape[-1] // patch_size * patch_size
    
    for i in tqdm(range(0, max_height, patch_size), desc="Getting patch indicies"):
        for j in range(0, max_width, patch_size):
            patch = raster[..., i:i+patch_size, j:j+patch_size]
            if validator(patch):
                patch_indicies.append((i, j))
                
    return patch_indicies

def plot_patch_extent(raster: np.ndarray, 
                      patch_indicies: list[tuple[int, int]], 
                      patch_size: int = 32) -> None:
    
    """Given a raster with shape (bands, height, width) and a list of patch indicies, draws a mask with shape (height, width) where the
    patchs in patch_indicies are white.

    Args:
        raster (np.ndarray): A raster stack of shape (bands, height, width)
        patch_indicies (list[tuple[int, int]]): List of tuples like (i, j) where i and j are the indicies of the top left corner of the patch
        patch_size (int, optional): Size of the patch. Defaults to 32.
    """
    mask = np.zeros_like(raster[0])
    for i, j in patch_indicies:
        mask[i:i+patch_size, j:j+patch_size] = 1
        
    plt.imshow(mask, cmap='gray')
    plt.show()
   
def save_patches(data: np.ndarray, 
                 patch_size: int, 
                 output_dir: str, 
                 output_type: str = 'npy',
                 verbose: bool = False) -> None:
    """Save patches from a numpy array to a directory.
    Output patchs are saved as numpy arrays with the format patch_i_j.npy where i and j are the indicies of the top left corner of the patch.
    
    Args:
        data (np.ndarray): A numpy array of shape where the last two dimensions are the height and width of the image. Supports any number of leading dimensions (bands, bands and time, etc.).
        patch_size (int): The size of the patches to save.
        output_dir (str): The directory to save the patches to
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose:
        print(f"Saving patches to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    max_height = data.shape[-2] // patch_size * patch_size
    max_width = data.shape[-1] // patch_size * patch_size
    
    for i in tqdm(range(0, max_height, patch_size), desc="Saving patches"):
        for j in range(0, max_width, patch_size):
            patch = data[..., i:i+patch_size, j:j+patch_size]

            patch_path = os.path.join(output_dir, f"{i}_{j}.{output_type}")
            save_raster_data(patch, patch_path)

def save_patches_from_labels(data: np.ndarray, 
                             output_dir: str, 
                             masks_dir: str, 
                             output_type: str = 'npy',
                             verbose: bool = False) -> None:
    """Save patches from a numpy array using an existing directory of masks. Patch size is inferred from the masks.

    Args:
        data (np.ndarray): A numpy array of shape where the last two dimensions are the height and 
        width of the image. Supports any number of leading dimensions (bands, bands and time, etc.).
        output_dir (str): The directory to save the patches to
        masks_dir (str): The directory with .npy patchs that contain the masks.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    assert os.path.exists(masks_dir), f"Masks directory {masks_dir} does not exist"
    assert len(os.listdir(masks_dir)) > 0, "Masks directory is empty"
    assert os.listdir(masks_dir)[0].endswith('.npy'), "Masks must be saved as .npy files"
    
    #test load one mask
    mask_patch = np.load(os.path.join(masks_dir, os.listdir(masks_dir)[0]))
    patch_size = mask_patch.shape[-2:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if verbose:
        print(f"Saving {len(os.listdir(masks_dir))} patches to {output_dir} using masks in {masks_dir}")
        
    for filename in os.listdir(masks_dir):
        i, j = map(int, filename.split('.')[0].split('_'))

        patch = data[..., i:i+patch_size, j:j+patch_size]
        full_path = os.path.join(output_dir, f"{i}_{j}.{output_type}")
        save_raster_data(patch, full_path)
