# Import necessary libraries for image manipulation and deep learning
import numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Install required libraries
!pip install natsort

# Mount Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

# Directory containing the image dataset
dir_path = '/content/drive/MyDrive/vehicle_dataset/images/'

# Opening and displaying a single image to verify correct loading
image_open = Image.open('/content/drive/MyDrive/vehicle_dataset/images/Rickshaw/images/rickshaw (54).jpg')
plt.imshow(image_open)
plt.title('Image : Rickshaw 53')
plt.show()

# Define the directory path and categories for the dataset
Train_dir = '/content/drive/MyDrive/vehicle_dataset/images/'
Categories = ['Bicycle/images','Bus/images','Car/images','Cng/images','Rickshaw/images','Truck/images']

# Loop through each category and load an image for visualization
for i in Categories:
    path = os.path.join(Train_dir, i)
    for img in os.listdir(path):
        old_image = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)  # Read image
        new_image = cv2.resize(old_image, (256, 256))  # Resize image to 256x256
        plt.imshow(old_image)  # Display the original image
        plt.show()
        break  # Stop after showing the first image in each category
    break  # Stop after processing the first category

# Show resized image
new_image = cv2.resize(old_image, (256, 256))  # Resize to 256x256
plt.imshow(new_image)  # Display resized image
plt.show()

# Importing necessary library for image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Install/upgrade TensorFlow to ensure the latest version
!pip install --upgrade tensorflow

# Define ImageDataGenerator for image augmentation (rescaling, shearing, zoom, flipping)
datagen = ImageDataGenerator(
    rescale=1/255,  # Normalize pixel values between 0 and 1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2  # Reserve 20% of data for validation
)

# Prepare the training data using the flow_from_directory method
train_datagen = datagen.flow_from_directory(
    Train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=16,  # Batch size for training
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Use training subset
)

# Prepare the validation data using the flow_from_directory method
val_datagen = datagen.flow_from_directory(
    Train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=16,  # Batch size for validation
    class_mode='categorical',  # Multi-class classification
    subset='validation'  # Use validation subset
)

# Display the class indices (mapping of labels to class names)
train_datagen.class_indices

# Import necessary components for building the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

# Define a simple CNN model architecture
model1 = Sequential()
model1.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))  # Conv layer with 64 filters
model1.add(MaxPooling2D(3, 3))  # Max pooling layer
model1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Conv layer with 32 filters
model1.add(MaxPooling2D(2, 2))  # Max pooling layer
model1.add(Conv2D(16, (3, 3), activation='relu', padding='same'))  # Conv layer with 16 filters
model1.add(MaxPooling2D(2, 2))  # Max pooling layer
model1.add(BatchNormalization())  # Normalize the activations

# Add dropout for regularization
model1.add(Dropout(0.1))
model1.add(Flatten())  # Flatten the feature map into a 1D vector
model1.add(Dense(64, activation='relu'))  # Fully connected layer with 64 neurons
model1.add(Dense(32, activation='relu'))  # Fully connected layer with 32 neurons
model1.add(Dense(6, activation='softmax'))  # Output layer with 6 classes (softmax for multi-class classification)
model1.summary()  # Display the model summary

# Define another CNN model with a different architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))  # Conv layer with 64 filters
model.add(MaxPooling2D(2, 2))  # Max pooling layer
model.add(BatchNormalization())  # Normalize the activations
model.add(Dropout(0.1))  # Dropout layer

# Adding more Conv layers and pooling
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Adding more Conv layers with increased filters
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Flatten, fully connected layers, and output layer
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 6 classes for classification
model.summary()  # Display the model summary

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Setup callbacks for saving best model and early stopping
checkpoint = ModelCheckpoint('vehicale_classifier.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, restore_best_weights=True)

callbacks = [checkpoint, earlystop]

# Train the model with training and validation data
model_history = model.fit(train_datagen, validation_data=val_datagen, epochs=30, callbacks=callbacks)

# Infinite loop to prevent the script from stopping (useful when running in certain environments)
while True:
    pass

# Plot the training and validation accuracy
train_acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
epoachs = range(1, 20)
plt.plot(epoachs, train_acc, 'g', label='Training Accuracy')
plt.plot(epoachs, val_acc, 'b', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the training and validation loss
loss_train = model_history.history['loss']
loss_val = model_history.history['val_loss']
epochs = range(1, 20)
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, loss_val, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save the trained model
model.save('vehicale_classifier.h5')

# Load and preprocess a test image for prediction
test_image = '/content/drive/MyDrive/vehicle_dataset/Test/test_2.jpg'
image_result = Image.open(test_image)

from tensorflow.keras.preprocessing import image
test_image = image.load_img(test_image, target_size=(256, 256))  # Load and resize the image
test_image = image.img_to_array(test_image)  # Convert image to array
test_image = test_image / 255  # Normalize the image
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Make prediction on the test image
result = model.predict(test_image)
print(np.argmax(result))  # Print the class with the highest probability

# Define categories corresponding to the 6 classes
Categories = ['Bicycle', 'Bus', 'Car', 'Cng', 'Rickshaw', 'Truck']

# Display the test image and predicted category
image_result = plt.imshow(image_result)
plt.title(Categories[np.argmax(result)])  # Title with predicted category
plt.show()

# Import TensorFlow and necessary applications for pre-trained models
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201, EfficientNetB7

# Import SVG for model visualization (not used in this code but can be helpful for visualization)
from IPython.display import SVG

# Install the EfficientNet library if it's not already installed
!pip install efficientnet

# Import EfficientNet from the EfficientNet TensorFlow Keras library
import efficientnet.tfkeras as efn

# Import necessary Keras layers for building the model
from keras.layers import Conv2D, MaxPooling2D

# Define the model using EfficientNetB7 as the base pre-trained model
model_efficientnet_b7 = tf.keras.Sequential([
    # Use EfficientNetB7 as the base model with ImageNet weights
    efn.EfficientNetB7(
        input_shape=(256, 256, 3),  # Input shape for the images (256x256x3)
        weights='imagenet',  # Load pre-trained weights from ImageNet
        include_top=False  # Exclude the top classification layer (we will add our own)
    ),
    
    # Apply max-pooling to reduce the spatial dimensions
    MaxPooling2D(pool_size=(2, 2)),  
    
    # Flatten the pooled feature maps into a 1D vector for input to fully connected layers
    Flatten(),
    
    # Add Batch Normalization to stabilize and speed up training
    BatchNormalization(),
    
    # Add Dropout layer to avoid overfitting during training (drop 30% of the units)
    Dropout(0.3),
    
    # Add a Dense (fully connected) layer with 9 neurons and softmax activation for multi-class classification
    Dense(9, activation='softmax')
])

# Compile the model using the Adam optimizer and categorical crossentropy as the loss function
model_efficientnet_b7.compile(
    optimizer='adam',  # Optimizer for training
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metric to monitor during training
)

# Set up the ModelCheckpoint callback to save the best model based on validation loss
checkpoint = ModelCheckpoint(
    r'vehicale_classifier.h5',  # Path where the best model will be saved
    monitor='val_loss',  # Monitor validation loss for best model selection
    mode='min',  # The lower the validation loss, the better the model
    save_best_only=True,  # Only save the best model (not every epoch)
    verbose=1  # Verbosity level to display information about the saving process
)

# Set up EarlyStopping callback to stop training early if validation loss doesn't improve
earlystop = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.001,  # Minimum change to qualify as an improvement
    patience=10,  # Number of epochs with no improvement after which training will stop
    verbose=1,  # Display information about early stopping
    restore_best_weights=True  # Restore the best model weights after stopping
)

# Combine the callbacks into a list
callbacks = [checkpoint, earlystop]

# Train the model on the training data using the ImageDataGenerator
model_history = model.fit(
    train_datagen,  # Training data generator
    validation_data=val_datagen,  # Validation data generator
    epochs=20,  # Number of epochs to train the model
    callbacks=callbacks  # Include the callbacks (checkpointing and early stopping)
)

# Infinite loop to prevent the script from stopping (useful for certain environments, but generally avoid in production code)
while True:
    pass

