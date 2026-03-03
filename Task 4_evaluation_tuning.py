# %% [markdown]
# # Step 1: Import Required Libraries
# For Evaluation and Hyperparameter Tuning

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# # Step 2: Plot Loss & Accuracy vs. Epoch
# Create a function to plot the training history.
# This helps us visually detect Overfitting or Underfitting.

# %%
def plot_training_history(history):
    """
    Plots the Accuracy and Loss graphs from a trained Keras model history.
    """
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', marker='o')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', marker='o')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red', marker='o')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='orange', marker='o')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Step 3: Regularization (Fixing Overfitting)
# If the Train Accuracy is high but Val Accuracy is low, the model is OVERFITTING.
# We fix this by adding Dropout and L2 Regularization.

# %%
def build_regularized_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001):
    """
    Builds a CNN model with Regularization applied.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        # Adding L2 Regularization (Weight Decay) to Dense layer
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        
        # Adding Dropout (turns off 50% of neurons randomly)
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# %% [markdown]
# # Step 4: Hyperparameter Tuning
# We will test different Batch Sizes and Learning Rates to find the best combo.

# %%
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    A simple grid search for tuning Learning Rate and Batch Size.
    """
    learning_rates = [0.001, 0.0001]
    batch_sizes = [16, 32]
    epochs = 5  # Keeping it small for tuning speed
    
    best_accuracy = 0
    best_params = {}
    
    print("Starting Hyperparameter Tuning...")
    
    for lr in learning_rates:
        for batch in batch_sizes:
            print(f"\nTraining with Learning Rate: {lr}, Batch Size: {batch}")
            
            # 1. Build Model
            model = build_regularized_model(learning_rate=lr)
            
            # 2. Train Model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch,
                verbose=1
            )
            
            # 3. Evaluate results
            val_acc = max(history.history['val_accuracy'])
            print(f"Max Validation Accuracy for (LR={lr}, Batch={batch}): {val_acc:.4f}")
            
            # 4. Save best parameters
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch}
                
    print("\n=========================================")
    print(f"Best Hyperparameters Found: {best_params}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print("=========================================")
    
    return best_params

# %% [markdown]
# # Step 5: Generate Confusion Matrix and Classification Report
# Evaluate the model purely on the Test Set (unseen data)

# %%
def evaluate_model_performance(model, X_test, y_test, class_names=None):
    """
    Generates a Confusion Matrix and Classification Report.
    """
    # 1. Get model predictions
    print("Generating predictions on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels
    
    # 2. Print Classification Report
    print("\n--- Classification Report ---")
    if class_names:
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    else:
        print(classification_report(y_test, y_pred, zero_division=0))
        
    # 3. Generate and Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    if not class_names:
        class_names = [str(i) for i in range(len(np.unique(y_test)))]
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# %% [markdown]
# # Example Usage Code (Commented Out)
# How to run all of this using your dataset from previous steps:

# %%
# # 1. Get the Best Hyperparameters
# best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

# # 2. Build final model using best parameters
# final_model = build_regularized_model(learning_rate=best_params['learning_rate'])

# # 3. Train final model
# final_history = final_model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=15, 
#     batch_size=best_params['batch_size']
# )

# # 4. Plot Loss and Accuracy to verify Overfitting has stopped
# plot_training_history(final_history)

# # 5. Evaluate final model on Test Data
# # class_names = ['Background', 'License Plate']
# # evaluate_model_performance(final_model, X_test, y_test, class_names=class_names)
