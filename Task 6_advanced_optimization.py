# %% [markdown]
# # Step 1: Import Required Libraries
# For Advanced Optimization and Transfer Learning

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2 # We use MobileNetV2 as it's fast and highly accurate
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# %% [markdown]
# # Step 2: Implement Transfer Learning (MobileNetV2)
# Here we will download a Pre-Trained model (MobileNetV2 trained on ImageNet).
# We will "freeze" the base layers so we don't destroy the pre-trained weights,
# and attach our own custom Dense layers at the end for our License Plate task.

# %%
def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Builds a Transfer Learning model using MobileNetV2 as the base.
    """
    print("Downloading MobileNetV2 Base Model...")
    
    # 1. Load the Base Model without the top prediction layer (include_top=False)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # 2. Freeze the base model layers (so they don't update during initial training)
    base_model.trainable = False 
    
    # 3. Add Custom Top Layers for our specific task (License Plate vs Background)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Better than Flatten() for Transfer Learning
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Final Output Layer
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 4. Combine into a new Model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 5. Compile Model
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("\nTransfer Learning Model built successfully!")
    return model

# %% [markdown]
# # Step 3: Compare Models & Save Best Model (.h5)
# This snippet provides the exact code needed to compare your Task 3 Custom CNN vs Task 6 MobileNet.
# It checks both Validation Accuracies and saves the ultimate winner as a `.h5` file.

# %%
def train_and_compare_models(custom_model, tl_model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Trains both models, compares their validation accuracy, and saves the better one.
    """
    print("========== TRAINING CUSTOM CNN (Task 3) ==========")
    custom_history = custom_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print("\n========== TRAINING TRANSFER LEARNING MODEL (Task 6) ==========")
    tl_history = tl_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Extract Best Validation Accuracies
    custom_best_acc = max(custom_history.history['val_accuracy'])
    tl_best_acc = max(tl_history.history['val_accuracy'])
    
    print("\n================= RESULTS =================\n")
    print(f"Custom Model (Task 3) Top Val Accuracy:  {custom_best_acc * 100:.2f}%")
    print(f"MobileNet TL (Task 6) Top Val Accuracy:  {tl_best_acc * 100:.2f}%")
    
    # Select best model
    if tl_best_acc > custom_best_acc:
        print("\n🏆 Transfer Learning Model WINS!")
        best_model = tl_model
        save_path = 'best_model_mobilenet.h5'
    else:
        print("\n🏆 Custom CNN Model WINS!")
        best_model = custom_model
        save_path = 'best_model_custom_cnn.h5'
        
    # Optional Fine-Tuning Step for Transfer Learning (Only if it won)
    if tl_best_acc > custom_best_acc:
        print("\nUnfreezing base model layers for fine-tuning...")
        # Unfreeze base model and retrain top layers at very low learning rate
        best_model.layers[0].trainable = True
        best_model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        best_model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))
        
    print(f"\n💾 Saving Best Model to disk as '{save_path}' for deployment...")
    
    # Save Model to .h5 for Task 5 Frontend
    best_model.save(save_path)
    
    print("Saved Successfully. You can now load this file in Task 5 Streamlit App!")
    return best_model

# %% [markdown]
# # Example Usage Code (Commented Out)
# Run the pipeline combining Task 3 (Custom Model) & Task 6 (Transfer Learning).

# %%
# # 1. Get your compiled Custom Model from Task 3
# custom_cnn = build_regularized_model() # (From your Task 4 script)

# # 2. Get Transfer Learning Model
# mobilenet_model = build_transfer_learning_model()

# # 3. Train, Compare, and Save Best
# # Assuming you have X_train, y_train, X_val, y_val loaded from Task 2 preprocessing
# best_model = train_and_compare_models(custom_cnn, mobilenet_model, X_train, y_train, X_val, y_val, epochs=10)
