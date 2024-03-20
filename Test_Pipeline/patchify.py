import cv2
import numpy as np
import os

def extract_patches(img, patch_size):
    patches = []
    h, w = img.shape[:2]
    ph, pw = patch_size
    for i, y in enumerate(range(0, h, ph)):
        for j, x in enumerate(range(0, w, pw)):
            patch = img[y:y+ph, x:x+pw]
            patches.append(patch)
    return patches

file_path = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/0'
for file in os.listdir(file_path):
    # Open the image
    image = cv2.imread(os.path.join(file_path,file))  # Replace "image.jpg" with your image file path

    # Extract patches of size 32x32
    patches = extract_patches(image, (32, 32))

    # Convert patches to numpy array
    patches_array = np.array(patches)

    # Save patches as .npy file
    np.save(os.path.join('/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/patches/0',file), patches_array)
