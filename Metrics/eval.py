import argparse
from cleanfid import fid

from base_dataset import BaseDataset
from utils import inception_score, perceptual_distance


import skimage.measure as measure
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import os

def compare_folders(folder1, folder2):
    ssim_sum = 0
    num_images = 0

    # Iterate through images in folder1
    for filename in os.listdir(folder2):
        if filename.endswith('.png'):
            image1_file = os.path.join(folder1, filename)
            image2_file = os.path.join(folder2, filename)

            # Load images
            image1 = Image.open(image1_file).convert('L')
            image2 = Image.open(image2_file).convert('L')

            # Convert images to arrays
            image1_arr = np.array(image1)
            image2_arr = np.array(image2)

            # Calculate structural similarity index
            ssim1 = ssim(image1_arr, image2_arr)

            # Update sum and count
            ssim_sum += ssim1
            num_images += 1

    # Calculate average SSIM
    avg_ssim = ssim_sum / num_images
    print("Num Images :", num_images)

    return avg_ssim



if __name__ == '__main__':

    # Correctly mention the source and destination folders.

    root_folder = 'test_latest/images_final/'
    real_A = 'real_A'
    real_B = 'real_B'
    fake_B = 'fake_B'

    print("Pix2Pix : ")
    
    # # Metric FID
    fid_score = fid.compute_fid(root_folder + real_B, root_folder + fake_B, device='cpu', num_workers=0)
    print('FID: {}'.format(fid_score))

    ssim = compare_folders(root_folder + real_B, root_folder + fake_B)
    print('ssim = {}'.format(ssim))

    print("Diffusion : ")

    diff_folder = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/Metrics/diff_test_resized"

    fid_score = fid.compute_fid(root_folder + real_B, diff_folder, device='cpu', num_workers=0)
    print('FID: {}'.format(fid_score))

    ssim = compare_folders(root_folder + real_B, diff_folder)
    print('ssim = {}'.format(ssim))

    print("Cycle GAN : ")

    diff_folder = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/eval_files_CycleGANs"

    fid_score = fid.compute_fid(root_folder + real_B, diff_folder, device='cpu', num_workers=0)
    print('FID: {}'.format(fid_score))

    ssim = compare_folders(root_folder + real_B, diff_folder)
    print('ssim = {}'.format(ssim))


