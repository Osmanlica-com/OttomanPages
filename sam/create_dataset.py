import numpy as np
import matplotlib.pyplot as plt
import os
from patchify import patchify  # Only to handle large images
import random
from scipy import ndimage
from datasets import Dataset, IterableDataset
from PIL import Image
import cv2

dataset_dir = "/workspace/masked_dataset"
large_images = [os.path.join(dataset_dir, "train", img) for img in os.listdir(os.path.join(dataset_dir, "train"))]
large_masks = [os.path.join(dataset_dir, "train_masked", img) for img in os.listdir(os.path.join(dataset_dir, "train_masked"))]

# Desired patch size for smaller images and step size.
patch_size = 256
step = 256

# # Directory to save the patches
patches_save_dir = "/workspace/patches_dataset"

all_img_patches = []

for img in range(len(large_images)):
    print(f"{img}/{len(large_images)}")
    large_image = cv2.imread(large_images[img], cv2.IMREAD_GRAYSCALE)
    patches_img = patchify(large_image, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            img_patch_path = os.path.join(patches_save_dir, f"image_{img}_{i}_{j}.png")
            Image.fromarray(single_patch_img).save(img_patch_path)

all_mask_patches = []

for img in range(len(large_images)):
    print(f"{img}/{len(large_masks)}")
    large_mask = cv2.imread(large_masks[img], cv2.IMREAD_GRAYSCALE)
    patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,j,:,:]
            single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
            mask_patch_path = os.path.join(patches_save_dir, f"mask_{img}_{i}_{j}.png")
            Image.fromarray(single_patch_mask).save(mask_patch_path)

image_paths = [os.path.join(patches_save_dir, img) for img in os.listdir(patches_save_dir) if img.startswith("image_")]
mask_paths = [os.path.join(patches_save_dir, img) for img in os.listdir(patches_save_dir) if img.startswith("mask_")]

def data_generator():
    for img_path, mask_path in zip(image_paths, mask_paths):
        mask = Image.open(mask_path)
        image = Image.open(img_path)
        mask_array = np.array(mask)
        if mask_array.max() != 0:
            yield {
                "image": image,
                "label": mask,
            }

# Load the dataset from disk
dataset = Dataset.from_generator(data_generator)
dataset.save_to_disk('/workspace/dataset_bulk')