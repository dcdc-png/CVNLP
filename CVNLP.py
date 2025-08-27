import warnings
warnings.filterwarnings('ignore') 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import colorama
from colorama import Fore, Style
from sklearn.model_selection import train_test_split


Root_dir = "C:/Users/limdi/Desktop/School/CVNLP/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/valid"
test_dir = "C:/Users/limdi/Desktop/CVNLP/test/test"

# Define the specific classes to include
Diseases_classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

print(Fore.GREEN + str(Diseases_classes))
print("\nTotal number of selected classes are: ", len(Diseases_classes))

# Set the number of images to display per class
subset_size = 10  
cols = 10  # Number of columns for each class grid
rows = len(Diseases_classes)  # Rows per class

# Create a figure to hold all images from each class
fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2), dpi=100) # Creates a space for the image to be displayed, and adjust the size and quality of image

# Loop through each class and plot 10 images in a grid
tot_images = 0
for class_idx, class_name in enumerate(Diseases_classes):
    image_files = os.listdir(train_dir + "/" + class_name) # Loops through each directory 
    sampled_files = random.sample(image_files, min(subset_size, len(image_files)))  # Randomly sample 10 images of each directory, scan through the total number of images in each files
    print("\nThe number of images in " + class_name + ": " , len(image_files), end=" ")
    tot_images += len(image_files)

    # Display each image with labels
    for i, img_file in enumerate(sampled_files):
        img_path = os.path.join(train_dir, class_name, img_file) # Holds the main directory, class name, and specific image name ex.(plant1.jpg)
        img_show = plt.imread(img_path)                          # Used to display the specific image
        
        # Display the image in the grid
        ax = axes[class_idx, i] # Position the images to row and column
        ax.imshow(img_show)     # Display the images
        ax.axis('off')  # Hide axis ticks and labels
        ax.set_title(class_name, fontsize=8)  # Set the label as the class name

print("\nThe total images inside the directory: ", tot_images , "\n")
plt.tight_layout()      # Adjust space between each images (Tidy up the text, images, and labels)
plt.suptitle("10 Sampled Images per Class", fontsize=20)
plt.show()

# MODEL ARCHITECTURE (KERNEL FILTER & MAX POOLING)
model = models.Sequential() # A sequential model, where user can add one layer to another
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # First Convolutional Layer with 32 number of filter with 3x3 matrix || relu helps the model to learn complex pattern || Shape of the input data 32 pixel high and wide with 3 colours (RGB)
model.add(layers.MaxPooling2D((2, 2))) # Adds a max pooling layer || Take maximum value of 2x2 region in the feature map
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Second Convolutional Layer with 64 number of filter to learn complex features || Doesn't need to put input shape cuz it already incur in previous layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Same as before to learn more complex feature || Activation relu is to set negative value to zero and postive remains unchanged
model.summary() # Print the summary of the model architecture

# FULLY CONNECTED LAYER
model.add(layers.Flatten()) # Flatten the image into column vector
model.add(layers.Dense(64, activation = 'relu')) # After column vector made, Dense Layer is connected to column vector to learn complex data and patterns
model.add(layers.Dense(10)) # Final output, represents the 10 classes in CIFAR-10

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),                # Shows how the model learn the data, to minimize error and improve training accuracy || Smaller learning rate shows indicates slower but stable learning
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),    # To minimize the loss predictions
              metrics=['accuracy'])                                                    # Tracks the accuracy during training and validation
# Loss measures the difference between predicted values and actual value || Accuracy is the percentage of the correct predictions

# Define image data generators (To avoid overfitting)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # Normalize the input to provide a better convergence (Allows machine to learn effectively, and provide better performance)
    rotation_range=20,         # Rotate the images to recognize the pattern in different angle to improve predictions
    width_shift_range=0.2,     # Shift the image horizontally for machines to recognize models in different parts
    height_shift_range=0.2,    # Shift the image  vertically for machines to recognize models in different parts
    shear_range=0.2,           # Apply 20% of slanting or tilting effect to improve the predictions
    zoom_range=0.2,            # Zoom the images to allow the model to learn at different scale
    horizontal_flip=True,      # Randomly flips the images horizontally, to view the images in different orientations to enhance the predictions
    fill_mode='nearest'        # Fill in blank pixesls in images, to avoid models to learn unnatural parts
)

# Collect a data from directories images and assign labels to each
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),       # Resize image to 32x32 pixels || Able to resize any images with different size, making machine to train consistently
    batch_size=32,              # Define how many images to be loaded per batch || Maximize computational efficiency and reducing idle time
    class_mode='sparse',        # Assign unique integer label to each images
    classes=Diseases_classes,   # From the three classes
    shuffle=True                # Shuffles the order of images to avoid patterns in order data that could affect training
)

# Train valid images
valid_datagen = ImageDataGenerator(rescale=1.0/255) # Rescale the picture to [0, 1] || Helps the model to train faster and more accurate by keeping input value small and consistent
valid_generator = valid_datagen.flow_from_directory( # Sets up images in valid directory to train images
    valid_dir,
    target_size=(32, 32),       
    batch_size=32,
    class_mode='sparse',
    classes=Diseases_classes,
    shuffle=False                   # Shuffling in valid directory is unnecessary, as the order of images doesnt impact the evaluation phase
)

# Train the model
epochs = 10                             # Sets the number of training cycle
history = model.fit(                    # Trains the model using specified data inside the fit() parameter
    train_generator,                    # Train images and assign labels
    epochs=epochs,                      # Undergoes training cycles
    validation_data=valid_generator     # Performs validation data
)

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))                                                # Specify the width and height of the graph
plt.subplot(1, 2, 1)                                                       # Create a grid with 1 row and 2 column, for placing the graph || Last digits shows the graph placing in the left side
plt.plot(history.history['accuracy'], label='Train Accuracy')              # Plots the training accuracy inside the graph
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')     # Plots the validation accuracy inside the graph
plt.title('Model Accuracy')                                                # Title of the graph
plt.ylabel('Accuracy')                                                     # y-axis label title
plt.xlabel('Epoch')                                                        # x-axis label title
plt.legend(loc='lower right')                                              # Legends located at the lower right

plt.subplot(1, 2, 2)                                                       # Last parameter, "2" shows that this graph will be placing at the right side
plt.plot(history.history['loss'], label='Train Loss')                      # Plot the result of train loss
plt.plot(history.history['val_loss'], label='Validation Loss')             # Plot the result of validation loss
plt.title('Model Loss')                                                    # Title of the graph
plt.ylabel('Loss')                                                         # y-axis label title
plt.xlabel('Epoch')                                                        # x-axis label title
plt.legend(loc='upper right')                                              # Legends located at upper right

plt.tight_layout()                                                         # Adjust the space between each graph (making the text, line, graph more tidy)
plt.show()                                                                 # Display the graph
