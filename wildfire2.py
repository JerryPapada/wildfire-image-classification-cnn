from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import pandas as pd

from telegram import Bot
import asyncio

async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

#====================================================== GPU =================================================

for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))

#======================================== DATA ACQUISITION & PREPROCESSING ===================================

# Define directories
train_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/train"
valid_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/valid"
test_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/test"

image_shape = (350,350,3)
N_CLASSES = 2
BATCH_SIZE = 256

# loading training data and rescaling it using ImageDataGenerator
train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical',
                                                   shuffle=True)

# loading validation data and rescaling it using ImageDataGenerator
valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
													batch_size = BATCH_SIZE,
													target_size = (350,350),
													class_mode = 'categorical',
                                                    shuffle=True)

# loading test data and rescaling it using ImageDataGenerator
test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_dir,
												  batch_size = BATCH_SIZE,
												  target_size = (350,350),
												  class_mode = 'categorical',
                                                  shuffle=True)

#=========================================== MODEL ARCHITECURE =====================================

# defining the coefficient that our regularizer will use
weight_decay = 1e-3

# building a sequential CNN model and adding layers to it
# dropout and the regularizer are used in general to prevent overfitting
first_model = Sequential([
    Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 32 , kernel_size = 2, activation = 'relu',
           kernel_regularizer = regularizers.l2(weight_decay)),
    MaxPooling2D(pool_size = 2),
    
    Dropout(0.4),
    Flatten(),
    Dense(300,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='softmax')
])
# showing the summary of our model (layers and number of parameters)
summary = first_model.summary()

#==================================== PARAMETERS ==================================

# don't stop everything if an image didn't load correctly
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# checkpointer to save the model only if it improved
checkpointer = ModelCheckpoint('first_model.hdf5',verbose=1, save_best_only= True)
# early stopping to stop the training if our validation loss didn't decrease for (10) consecutive epochs
early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)
# Adam, best optimiser for deep learning models to help with the training
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001, decay=1e-5)
# setting our loss function and which metric to evaluate
first_model.compile(loss= 'categorical_crossentropy', optimizer= optimizer,metrics=['AUC','acc'])

# training
history = first_model.fit(train_generator,
                    epochs = 10,
                    verbose = 1,
                    validation_data = valid_generator,
                    callbacks = [checkpointer, early_stopping])


#====================================== PLOTS =======================================

# add history of accuracy and validation accuracy to the plot
plt.plot(history.history['acc'], label = 'train',)
plt.plot(history.history['val_acc'], label = 'valid')

# adding legend and labels
plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')

# show the plot
plt.show()

#======================================= TESTING ====================================

# see if it's good at predecting new inputs
result = first_model.evaluate(test_generator)
print("Result:", result)

# Evaluate the model on the test set
test_loss, test_auc, test_accuracy = first_model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test AUC:", test_auc)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test set
predictions = first_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels of the test set
true_classes = test_generator.classes

# Calculate additional metrics
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

# Generate a classification report
class_names = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("Classification Report:")
print(report)

ml_results = {
    #"summary": summary,
    "result": {
        "loss": test_loss,
        "AUC": test_auc,
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        #"classification_report": report
    }
}

async def main(ml_results):
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results)

if __name__ == "__main__":
    asyncio.run(main(ml_results))


