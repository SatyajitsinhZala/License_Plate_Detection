# %% [markdown]
# # Step 1: Import Required Libraries
# Import TensorFlow and Keras modules for Model Building:

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# %% [markdown]
# # Step 2: Design Custom Neural Network Architecture (CNN)
# We use a Convolutional Neural Network (CNN) as this is an image processing task.
# We will define layers: Conv2D, MaxPool, Flatten, Dense, and Dropout.

# %%
# Define image dimensions (matching the preprocessing step)
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

# Define number of classes (e.g., 2 for Plate vs. Background)
NUM_CLASSES = 2 

model = Sequential([
    # 1st Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 2nd Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 3rd Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Transition to Dense Layers
    Flatten(),
    
    # Fully Connected (Dense) Layer
    Dense(128, activation='relu'),
    
    # Dropout to prevent overfitting
    Dropout(0.5),
    
    # Output Layer 
    Dense(NUM_CLASSES, activation='softmax')
])

# Display the model architecture
model.summary()

# %% [markdown]
# # Step 3: Compile the Model
# Compile the model with an appropriate Optimizer (e.g., Adam) and Loss Function.

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use 'binary_crossentropy' if strictly binary and using sigmoid output
    metrics=['accuracy']
)

print("Model compiled successfully with Adam optimizer and Sparse Categorical Crossentropy loss.")

# %% [markdown]
# # Step 4: Train the Model
# Train the model for standard epochs.
# 
# *Note: Below is the code to run the training using the variables `X_train`, `y_train`, `X_val`, and `y_val` from Task 2.*

# %%
EPOCHS = 10
BATCH_SIZE = 32

print(f"Preparing to train model for {EPOCHS} epochs...")

# =========================================================================
# UNCOMMENT the code below when running alongside your Task 2 environment!
# =========================================================================

# history = model.fit(
#     X_train, 
#     y_train, 
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS, 
#     validation_data=(X_val, y_val)
# )

# # Visualize Training History (Accuracy and Loss)
# plt.figure(figsize=(12, 4))
#
# # Accuracy Plot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Loss Plot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss', color='red')
# plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
