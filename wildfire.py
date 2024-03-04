import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import os
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from telegram import Bot
import asyncio

async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

#============================================= PREPROCESSING & TRUNCATED IMG ==========================================

# Define a function to load and preprocess images, handling truncated images
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        return img_array, None
    except OSError as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None, e

# Define a function to delete truncated images
def delete_truncated_images(directory):
    deleted_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            image_path = os.path.join(root, file)
            try:
                img = Image.open(image_path)
                img.load()
            except (OSError, SyntaxError) as e:
                print(f"Deleting truncated image: {image_path}")
                os.remove(image_path)
                deleted_count += 1
    print(f"Deleted {deleted_count} truncated images.")

# Define directories
train_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/dfire/train"
#valid_dir = "/home/gerpapa/thesis/dataset/valid"
test_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/dfire/test"

# Delete truncated images in training directory
#delete_truncated_images(train_dir)

# Delete truncated images in validation directory
#delete_truncated_images(valid_dir)

# Delete truncated images in test directory
#delete_truncated_images(test_dir)

#============================================== MODEL ARCHITECTURE ================================================

# ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# ImageDataGenerator for validation and test data (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  
    batch_size=16,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# Flow validation images in batches of 32 using valid_datagen generator
valid_generator = valid_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

# Flow testing images in batches of 32 using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)

#==================================================================================================================
#============================================== HUPERPARAMETER TUNING =============================================

def build_model(hp):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(hp.Choice('units', [64, 128, 256], default=128), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
'''
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=2,   # USE THIS
    directory='hyperparameter_tuning',
    project_name='wildfire_cnn'
)
tuner.search(train_generator, validation_data=valid_generator, epochs=10)
'''
#========================================================================================================#
#======================================== LAYERS & TRAIN ================================================#

# Define the model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),  # Additional Convolutional Layer
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),  # Additional Convolutional Layer
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),  # Add dropout with a rate of 0.5
    Dense(128, activation='relu'),
    Dropout(0.3),
])


#model = build_model(HyperParameters())
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=valid_generator,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test Accuracy:", test_accuracy)

# Get true labels of the test set
test_true_labels = test_generator.classes

# Make predictions on the test set
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predictions = np.round(predictions).flatten()

# Ensure that predictions are trimmed to match the length of the true labels
predictions = predictions[:len(test_true_labels)]

# Print the length of predictions after trimming
print("Length of predictions after trimming:", len(predictions))

# Print lengths of true labels and predictions
print("Length of true labels:", len(test_true_labels))
print("Length of predictions:", len(predictions))


# Calculate evaluation metrics
accuracy = accuracy_score(test_true_labels, predictions)
precision = precision_score(test_true_labels, predictions)
recall = recall_score(test_true_labels, predictions)
f1 = f1_score(test_true_labels, predictions)

ml_results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}

async def main(ml_results):
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results)

if __name__ == "__main__":
    asyncio.run(main(ml_results))

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
