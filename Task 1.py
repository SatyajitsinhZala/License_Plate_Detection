# %% [markdown]
# # Step 1: Import Required Libraries
# Run this first cell:

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# %% [markdown]
# # Step 2: Define Project & Task
# 
# # Vehicle Log System (License Plate Recognition)
# 
# ## Problem Definition
# 
# This project aims to develop an automated Vehicle Log System 
# that detects license plates from vehicle images and extracts 
# the vehicle registration number using Deep Learning.
# 
# ## Type of Deep Learning Task
# 
# 1. Object Detection - Detect license plate region.
# 2. Optical Character Recognition (OCR) - Extract text from plate.
# 
# This is a Computer Vision based Object Detection + OCR system.

# %% [markdown]
# # Step 3: Load Dataset Path
# Load dataset path and count total images.

# %%
dataset_path = r"c:\Users\LENOVO\Downloads\archive\google_images"

# Count total images
image_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(root, file))

print("Total Images:", len(image_files))

# %% [markdown]
# # Step 4: Display Sample Images
# This shows dataset preview.

# %%
plt.figure(figsize=(10,6))

# We use min(5, len(image_files)) to avoid errors if there are less than 5 images
n_samples = min(5, len(image_files))

for i in range(n_samples):
    img = cv2.imread(image_files[i])
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, n_samples, i+1)
        plt.imshow(img)
        plt.axis("off")

plt.show()

# %% [markdown]
# # Step 5: Analyze Image Sizes (Data Quality)

# %%
widths = []
heights = []

for img_path in image_files[:100]:  # check first 100 images
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

if len(widths) > 0:
    print("Average Width:", np.mean(widths))
    print("Average Height:", np.mean(heights))
else:
    print("No images to analyze.")
