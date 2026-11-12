import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import hyperparameters as hp
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from telegram import Bot
import asyncio

#========================================================================================
# function to send results to my phone 
async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

# build model architecture function
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# compile model function
def compile_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy','AUC','recall'])

    return model

# train model function
def train_model(model, train_generator, validation_generator, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    results = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return results

# create data generators - split train,valid and test data
# data augmentation
def create_generators(train_dir, validation_dir, test_dir):
    # Train data generator with augmentation
    augmented_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.7, 1.1],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = augmented_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(hp.img_height, hp.img_width),
        color_mode="rgb",
        batch_size=hp.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42
    )

    # Validation data generator without augmentation
    validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=validation_dir,
        target_size=(hp.img_height, hp.img_width),
        color_mode="rgb",
        batch_size=hp.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42
    )

    # Test data generator without augmentation
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=test_dir,
        target_size=(hp.img_height, hp.img_width),
        color_mode="rgb",
        batch_size=hp.batch_size,
        class_mode="binary",
        shuffle=False,  # Set shuffle to False for test set to maintain order
        seed=42
    )

    return train_generator, validation_generator, test_generator


def evaluate_model(model, test_generator):
    # Evaluate the model on the test set
    evaluation_metrics = model.evaluate(test_generator)

    # Extract individual evaluation metrics
    test_loss = evaluation_metrics[0]
    test_accuracy = evaluation_metrics[1]
    test_auc = evaluation_metrics[2]

    print("Test Loss:", test_loss)
    print("Test AUC:", test_auc)
    print("Test Accuracy:", test_accuracy)

    # Make predictions on the test set
    predictions = model.predict(test_generator)
    
    # Get true labels of the test set
    true_classes = test_generator.classes

    # Convert predicted probabilities to predicted classes
    predicted_classes = np.where(predictions > 0.5, 1, 0)  # Assuming binary classification and threshold of 0.5

    # Calculate additional metrics
    precision = precision_score(true_classes, predicted_classes, zero_division=1)  # Set zero_division parameter to 1
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)

    # Generate a classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_names)
    
    ml_results = {
        "result": {
            "loss": test_loss,
            "AUC": test_auc,
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report
        }
    }

    return ml_results


async def main():
    #SATELLITE DATASET
    #train = "/home/gerpapa/Wildfire-Classification-CNN/dataset/train"
    #validation = "/home/gerpapa/Wildfire-Classification-CNN/dataset/valid"
    #test = "/home/gerpapa/Wildfire-Classification-CNN/dataset/test"

    #forest fire dataset
    train = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/train"
    validation = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/valid"
    test = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/test"
    
    #dfire
    #train = "/home/gerpapa/Wildfire-Classification-CNN/data/output/train"
    #validation = "/home/gerpapa/Wildfire-Classification-CNN/data/output/val"
    #test = "/home/gerpapa/Wildfire-Classification-CNN/data/output/test"

    train_generator, validation_generator, test_generator = create_generators(train, validation, test)

    model = build_model(input_shape=(hp.img_height, hp.img_width, 3))
    model = compile_model(model)
    
    results = train_model(model, train_generator, validation_generator, epochs=hp.epochs)
    ml_results = evaluate_model(model,test_generator)
    await send_telegram_message(os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"], ml_results)

    return results


if __name__ == "__main__":
    asyncio.run(main())