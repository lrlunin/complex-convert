"""Synthetic FastMRI Low-Field Dataset.

2020 fastMRI challenge (brain-data)
Training and validation data:
    - "k-space": multi-coil k-space data, dimensions = (slices, coils, height, width)
    - "reconstruction_rss": Root-sum-of-squares reconstruction of multi-coil k-space,
    cropped to center, dimension = (slices, heigh, width)

Dataset location: /home/global/mri_datasets/fastmri/brain_multicoil_train/
Total number of samples: 4469
"""
import logging
import pathlib
from collections.abc import Callable
import torch
import h5py

def convert_data(in_filepath: pathlib.Path,
                 file_index:int,
                 files_overall:int,
                 out_dir: pathlib.Path,
                 smooth_width:int,
                 proc_func:Callable,
                 use_cuda:bool):
    logger = logging.getLogger('convert_data()')
    logger.info(f"Started {in_filepath}")
    save_path = out_dir / in_filepath.name
    if save_path.exists():
        logger.info(f"Skip, {save_path} already exists")
        return
    # dimension of k_space is (slices, coils, height, width)
    try:
        k_space_raw = torch.tensor(h5py.File(in_filepath)["kspace"][:])
        # mrmpro.walsh expects data to be (coils, z, height (y), width (x))
        # since the slices are not lie on each other (not real z dimension)
        # we observe the slices individually
        # this results the shape of (slices, coils, 1, height, width)
        k_space = torch.unsqueeze(k_space_raw, 2)
        # put data to the GPU
        if use_cuda:
            k_space = k_space.to(device="cuda")
        image_space = torch.fft.ifftshift(
            torch.fft.ifftn(
            torch.fft.fftshift(k_space, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
        # evaluate for each slice in slices
        sl_list = [proc_func(sl, smooth_width) for sl in torch.unbind(image_space, 0)]
        # sensitivity maps for each slices (slices, coils, 1, height, width)
        sens_maps = torch.stack(sl_list, dim=0)
        # see mrpro.operators.SensitivityOp
        # applying the sensitivity maps to the image_space images
        # sum over coils and "1" dimension to obtain the normalized images
        # of shape (slices, height, width)
        result = torch.sum(sens_maps.conj()*image_space, dim=(1,2))
        if use_cuda:
            result = result.to(device='cpu')
        with h5py.File(save_path, mode="w") as save_file:
            save_file.create_dataset("reconstruction_smap", data=result)
            save_file.create_dataset("kspace", data=k_space_raw)
        logger.info(f"[{file_index+1} / {files_overall}] Sucessfully evaluated {in_filepath} to {save_path}")
    except torch.cuda.OutOfMemoryError as e:
        logger.critical(f"Error while converting {filepath}: {e}")
