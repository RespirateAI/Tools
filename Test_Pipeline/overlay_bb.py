import cv2
import numpy as np
import os
import random

def overlay_transparent(background, overlay, x, y):
    # Extract the alpha channel of the overlay image
    overlay_alpha = overlay[:, :, 3] / 255.0
    # Extract the inverse of the alpha channel
    background_alpha = 1.0 - overlay_alpha

    # Calculate the weighted sum of the alpha channels
    for c in range(0, 3):
        background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = (
            overlay_alpha * overlay[:, :, c] +
            background_alpha * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c]
        )
    return background

# Load background images
backgrounds_folder = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/Selected_bg"
background_images = os.listdir(backgrounds_folder)

# Load basketball image with transparent background
basketball_img = cv2.imread("/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/Basketball.png", cv2.IMREAD_UNCHANGED)

# Set the number of random positions for basketball image
num_positions = 20

# Define target sizes for basketball image
target_sizes = [(32, 32), (50, 50), (64, 64)]

# Loop through each background image
for background_image_name in background_images:
    background_path = os.path.join(backgrounds_folder, background_image_name)
    background = cv2.imread(background_path)

    # Resize background image to 256x256
    background_resized = cv2.resize(background, (256, 256))

    # Get the dimensions of the resized background image
    bg_height, bg_width = background_resized.shape[:2]

    # Loop through each target size
    for target_size in target_sizes:
        # Resize basketball image to target size
        basketball_resized = cv2.resize(basketball_img, target_size)

        # Get the dimensions of the basketball image
        basketball_height, basketball_width = basketball_resized.shape[:2]

        # Ensure basketball image fits within the resized background image
        if bg_height < basketball_height or bg_width < basketball_width:
            print(f"Skipping {background_image_name} with target size {target_size} as the basketball image does not fit within it.")
            continue

        # Loop through each random position
        for p in range(num_positions):
            # Generate random position within the resized background image
            x = random.randint(0, bg_width - basketball_width)
            y = random.randint(0, bg_height - basketball_height)

            # Overlay the basketball image on the resized background
            background_with_basketball = overlay_transparent(background_resized.copy(), basketball_resized.copy(), x, y)

            # Save the image with basketball to a new file
            output_filename = f"{os.path.splitext(background_image_name)[0]}_with_basketball_{target_size[0]}x{target_size[1]}_pos{p}.png"
            output_path = os.path.join("/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset", output_filename)
            cv2.imwrite(output_path, background_with_basketball)

print("Images with basketball overlay created successfully.")
