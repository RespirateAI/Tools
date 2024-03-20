import os
import cv2
import Augmentor

# Define paths
input_folder = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/Selected_bg"
output_folder = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/ground_truths"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to crop images into 256x256
def crop_image(image):
    height, width = image.shape[:2]
    start_x = (width - 256) // 2
    start_y = (height - 256) // 2
    cropped_image = image[start_y:start_y+256, start_x:start_x+256]
    return cropped_image

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Crop the image to 256x256
        cropped_image = crop_image(image)

        # Save the cropped image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped_image)

        # Create Augmentor pipeline for data augmentation
        p = Augmentor.Pipeline(output_folder)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
        p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
        p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)

        # Sample 10 random augmentations from the pipeline
        p.sample(10)

print("Images cropped and augmented successfully.")
