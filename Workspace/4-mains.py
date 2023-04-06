import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# set the input image size
IMG_SIZE = 350

def load_images_labels(data_path):
    # get all subdirectories of data_path. Each represents a class
    directories = list(filter(lambda f: os.path.isdir(os.path.join(data_path, f)), os.listdir(data_path)))
    
    images = []
    labels = []

    # iterate over every subdirectory (class)
    for i, directory in enumerate(directories):
        current_label = i
        
        # iterate over every file in the subdirectory (image)
        for filename in os.listdir(os.path.join(data_path, directory)):
            if not filename.endswith('.jpg'):
                continue
            
            # open the file using PIL and resize with respect to IMG_SIZE
            img_path = os.path.join(os.path.join(data_path, directory), filename)
            img = Image.open(img_path)
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            
            # Append the preprocessed image and label to the lists
            images.append(np.array(img_resized))
            labels.append(current_label)

    return images, labels


# Load MRI images and labels
data_path = '/kaggle/input'
images, labels = load_images_labels(data_path)

# shuffle the images and labels
zipped_data = list(zip(images, labels))
random.shuffle(zipped_data)
shuffled_imgs, shuffled_lbls = zip(*zipped_data)

# split the images and labels into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    shuffled_imgs,
    shuffled_lbls,
    test_size=0.2,
    random_state=42
)

# normalize the pixel values of the images
train_images = np.array(train_images) / 255.0
val_images = np.array(val_images) / 255.0

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with an Adam optimizer and binary cross-entropy loss
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the validation set
_, val_acc = model.evaluate(val_images, val_labels, verbose=0)
print('Validation Accuracy:', val_acc)

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()