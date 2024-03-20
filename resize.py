from PIL import Image
import os
import glob
import numpy as np

fire = "data/raw/fire_images/*.png"
non_fire = "data/raw/non_fire_images/*.png"

# Make directories for preprocessed data:

if not os.path.exists("/home/gerpapa/CNN-new/wildfire-cnn/data/processed/fire_images"):
    os.makedirs("/home/gerpapa/CNN-new/wildfire-cnn/data/processed/fire_images")
if not os.path.exists("/home/gerpapa/CNN-new/wildfire-cnn/data/processed/non_fire_images"):
    os.makedirs("/home/gerpapa/CNN-new/wildfire-cnn/data/processed/non_fire_images")
processed_fire = "/home/gerpapa/CNN-new/wildfire-cnn/data/processed/fire_images"
processed_non_fire = "/home/gerpapa/CNN-new/wildfire-cnn/data/processed/non_fire_images"

# Load and resize fire images
for image in glob.glob(fire):
    img = Image.open(image)

    orig_width, orig_height = img.size
    new_width, new_height = 200, 200
    img = img.crop(((orig_width - new_width) // 2,
                    (orig_height - new_height) // 2,
                    (orig_width + new_width) // 2,
                    (orig_height + new_height) // 2)
                  )
    image_name = os.path.basename(image)
    img.convert('RGB').save(os.path.join(processed_fire, image_name))

# Load and resize non-fire images
for image in glob.glob(non_fire):
    img = Image.open(image)

    orig_width, orig_height = img.size
    new_width, new_height = 200, 200
    img = img.crop(((orig_width - new_width) // 2,
                    (orig_height - new_height) // 2,
                    (orig_width + new_width) // 2,
                    (orig_height + new_height) // 2)
                  )
    image_name = os.path.basename(image)
    img.convert('RGB').save(os.path.join(processed_non_fire, image_name))

# Create training, validation, and testing sets
import splitfolders
splitfolders.ratio("/home/gerpapa/CNN-new/wildfire-cnn/data/processed", output="/home/gerpapa/CNN-new/wildfire-cnn/data/output", seed=1337, ratio=(0.8,0.1,0.1))