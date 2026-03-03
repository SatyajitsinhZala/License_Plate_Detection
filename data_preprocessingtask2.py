# %% [markdown]
# # Step 1: Import Required Libraries
# Let's import the necessary libraries for data preprocessing and augmentation:

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# # Step 2: Data Preprocessing
# 
# ## Images: Resize, Rescale, Normalization
# ## Text/Audio: Tokenization, Padding, Spectrogram conversion (Skipped: Project is Computer Vision / Images)

# %%
dataset_path = r"c:\Users\LENOVO\Downloads\archive\google_images"
image_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(root, file))

IMAGE_SIZE = (224, 224)  # Standard size for most CNNs like ResNet/VGG
X_data = []

print("Loading and Resizing images...")
# We will load a subset to prevent out-of-memory errors for this script, 
# for full training you may use generators instead of loading all into RAM.
for img_path in image_files[:150]: 
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Resize Images to uniform shape
        img_resized = cv2.resize(img, IMAGE_SIZE)
        X_data.append(img_resized)

X_data = np.array(X_data)
print("Data Shape before normalization:", X_data.shape)

# 2. Rescale & Normalize Images (scale pixel values to between 0 and 1)
X_data = X_data.astype('float32') / 255.0
print("Data Shape after normalization:", X_data.shape)
print(f"Min pixel: {np.min(X_data)}, Max pixel: {np.max(X_data)}")

# %% [markdown]
# # Step 3: Split into Train / Validation / Test sets
# Using 70% Train, 15% Validation, 15% Test.

# %%
# Using dummy labels for the sake of the structural split 
y_dummy = np.zeros(len(X_data)) 

# First split to separate out the 15% Test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_dummy, test_size=0.15, random_state=42
)

# Second split to separate the remaining 85% into Train (70%) and Validation (15%)
# 0.15 / 0.85 approx = 0.1765
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42
)

print(f"Total instances: {len(X_data)}")
print(f"Train set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# %% [markdown]
# # Step 4: Data Augmentation
# 
# Apply rotation, flipping, zooming to increase dataset size artificially.
# This prevents overfitting and helps the model generalize better.

# %%
# Define Augmentation Generator
datagen = ImageDataGenerator(
    rotation_range=20,     # Apply Rotation (±20 degrees)
    zoom_range=0.2,        # Zooming in/out by 20%
    horizontal_flip=True,  # Flipping horizontally
    width_shift_range=0.1, # Shift left/right
    height_shift_range=0.1,# Shift up/down
    fill_mode='nearest'    # Strategy to fill newly created pixels
)

# Fit generator on training data (only required for some features like ZCA whitening, but good practice)
datagen.fit(X_train)

# Visualize Original vs Augmented Images
plt.figure(figsize=(15, 5))

# Get a sample from our train set to augment
sample_image = X_train[0:1] # shape (1, 224, 224, 3)

plt.subplot(1, 4, 1)
plt.imshow(sample_image[0])
plt.title("Original Image")
plt.axis("off")

# Generate 3 augmented versions
aug_iter = datagen.flow(sample_image, batch_size=1)
for i in range(3):
    aug_img = next(aug_iter)[0]
    plt.subplot(1, 4, i+2)
    plt.imshow(aug_img)
    plt.title(f"Augmented Image {i+1}")
    plt.axis("off")

plt.show()

print("Data Preprocessing & Augmentation Pipeline completed successfully!")
