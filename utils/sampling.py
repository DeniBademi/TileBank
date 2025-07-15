"""
This module contains functions for sampling patches from raster images.
"""

import numpy as np
from tqdm import tqdm
import os

def sample_patches(data: np.ndarray, patch_size: int, max_patches: int, sampler: callable, verbose: bool = False) -> list[tuple[int, int]]:
    """Sample patches from a raster image.
    
    Args:
        data (np.ndarray): The data to sample patches from.
        patch_size (int): The size of the patches to sample.
        max_patches (int): The maximum number of patches to sample.
        sampler (callable): A function that takes a patch and returns True if the patch should be included in the dataset.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        list[tuple[int, int]]: A list of tuples containing the indicies of the patches.
    """
    
    indicies = []

    max_height = data.shape[-2] // patch_size * patch_size
    max_width = data.shape[-1] // patch_size * patch_size
    
    for i in tqdm(range(0, max_height, patch_size), desc="Sampling patches"):
        for j in range(0, max_width, patch_size):
            patch = data[..., i:i+patch_size, j:j+patch_size]
            if sampler(patch):
                indicies.append((i, j))
            
            if len(indicies) >= max_patches:
                if verbose:
                    print(f"Reached max patches: {max_patches}")
                return indicies
    if verbose:
        print(f"Sampled {len(indicies)} patches")

    return indicies

def sample_patches_in_dir(data_dir : str, patch_size : int, max_patches : int, sampler: callable, verbose : bool = False) -> list[tuple[str, int, int]]:
    """Sample patches from a directory of raster images.
    
    Args:
        data_dir (str): The directory containing the raster images.
        patch_size (int): The size of the patches to sample.
        max_patches (int): The maximum number of patches to sample from a single raster.
        sampler (callable): A function that takes a patch and returns True if the patch should be included in the dataset.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        list[tuple[str, int, int]]: A list of tuples containing the path to origin raster, the height and width index of the patch.
    """
    
    indicies = []
    
    for file in tqdm(os.listdir(data_dir), desc="Sampling patches"):
        raster = np.load(os.path.join(data_dir, file))
        raster_indicies = sample_patches(raster, patch_size, max_patches, sampler, verbose)
        indicies.extend([(file, i, j) for i, j in raster_indicies])
        
    return indicies
