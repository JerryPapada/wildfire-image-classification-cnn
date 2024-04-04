from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
from telegram import Bot
import asyncio

#====================================== TELEGRAM ==========================================

async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

#=============================== K-FOLD CROSS VALIDATTION ==================================

def k_fold_cross_validation(train_dir, valid_dir, num_classes, k=5, epochs=10, batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(train_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
    
    valid_generator = datagen.flow_from_directory(valid_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []
    for train_index, val_index in skf.split(train_generator.filenames, train_generator.classes):
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator_fold = train_datagen.flow_from_directory(train_dir,
                                                                 target_size=(img_height, img_width),
                                                                 batch_size=batch_size,
                                                                 class_mode='categorical',
                                                                 subset='training',
                                                                 shuffle=True,
                                                                 seed=42)
        
        val_generator_fold = val_datagen.flow_from_directory(train_dir,
                                                             target_size=(img_height, img_width),
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             subset='validation',
                                                             shuffle=True,
                                                             seed=42)

        input_shape = train_generator_fold.image_shape
        model = create_cnn_model(input_shape, num_classes)
        
        model.fit(train_generator_fold,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=val_generator_fold,
                  verbose=0)
        
        _, accuracy = model.evaluate(valid_generator, verbose=0)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f'Mean accuracy: {mean_accuracy:.2f}, Standard deviation: {std_accuracy:.2f}')

# Example usage:
# Assuming you have train_dir, valid_dir containing the paths to your training and validation data folders
# and num_classes indicating the number of classes in your classification task.
# You would call k_fold_cross_validation(train_dir, valid_dir, num_classes)


# Example usage:
# Assuming you have train_dir, valid_dir containing the paths to your training and validation data folders
# and num_classes indicating the number of classes in your classification task.
# You would call k_fold_cross_validation(train_dir, valid_dir, num_classes)


#============================= DATA ACQUISITION & PREPROCESSING =============================
image_shape = (224,224,3)
target_size = (224,224)

def preprocessing(batch_size, target_size):
    # Define directories
    train_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/train"
    valid_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/valid"
    test_dir = "/home/gerpapa/Wildfire-Classification-CNN/dataset/test"

    BATCH_SIZE = batch_size

    # loading training data and rescaling it using ImageDataGenerator
    train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.,
                                                        rotation_range=20,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,)
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = BATCH_SIZE,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    shuffle=True)

    # loading validation data and rescaling it using ImageDataGenerator
    valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        batch_size = BATCH_SIZE,
                                                        target_size = target_size,
                                                        class_mode = 'categorical',
                                                        shuffle=True)

    # loading test data and rescaling it using ImageDataGenerator
    test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size = BATCH_SIZE,
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    shuffle=True)

    generators = [train_generator, valid_generator, test_generator]
    return generators

#============================= MODEL ARCHITECURE ================================
def create_model(image_shape):
    # defining the coefficient that our regularizer will use
    weight_decay = 0.001

    # building a sequential CNN model and adding layers to it
    # dropout and the regularizer are used in general to prevent overfitting
    model = Sequential([
        Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', 
        input_shape = image_shape), MaxPooling2D(pool_size = 2),
        
        Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', 
        input_shape = image_shape), MaxPooling2D(pool_size = 2),
        
        Conv2D(filters = 32 , kernel_size = 2, activation = 'relu',
        kernel_regularizer = regularizers.l2(weight_decay)),MaxPooling2D(pool_size = 2),
        
        Dropout(0.5),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.8),
        Dense(2,activation='sigmoid')
    ])
    print(model.summary())
    return model

#================================== GRID SEARCH =================================
def grid_search():
    # grid search parameters
    epochs = [100, 200, 300, 500]
    batch_size = [16,32,64,128,256]
    image_shape = [(64,64,3),(128,128,3),(256,256,3),(512,512,3)]
    best_parameters = {}
    best_results = {}
    best_score = 0

    # grid search execution
    for epoch in epochs:
        for batch in batch_size:
            for size in image_shape:
                # create target size tuple
                target = size[:-1]

                print(f"Training model with image size: {size} ,batch size {batch}, epochs {epoch}...")
                generators = preprocessing(batch, target)
                model = create_model(size)
                history = train_model(model, generators)
                results = evaluate_model(model, generators)
                print(f"Results: {results}")

                if(results["result"]["accuracy"] > best_score):
                    best_score = results["result"]["accuracy"]
                    best_parameters = {"batch_size": batch, "epochs": epoch, "size": size}
                    best_results = results
    
    grid_results = {
        "best_parameters" : best_parameters,
        "results" : best_results
    }
    
    return grid_results

#==================================== TRAINING ==================================
def train_model(model, generators):
    train_generator = generators[0]
    valid_generator = generators[1]

    # checkpointer to save the model only if it improved
    checkpointer = ModelCheckpoint('cnn.hdf5',verbose=1, save_best_only= True)
    # early stopping to stop the training if our validation loss didn't decrease for (10) consecutive epochs
    early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)
    # Adam, best optimiser for deep learning models to help with the training
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001, decay=1e-5)
    # setting our loss function and which metric to evaluate
    model.compile(loss= 'binary_crossentropy', optimizer= optimizer,metrics=['AUC','acc'])

    # training
    history = model.fit(train_generator,
                        epochs = 10, # change this
                        verbose = 1,
                        validation_data = valid_generator,
                        callbacks = [checkpointer, early_stopping])
    
    return history

#===================================== TESTING ================================

def evaluate_model(model, generators):
    test_generator = generators[2]

    # see if it's good at predecting new inputs
    result = model.evaluate(test_generator)
    print("Result:", result)

    # Evaluate the model on the test set
    test_loss, test_auc, test_accuracy = model.evaluate(test_generator)
    print("Test Loss:", test_loss)
    print("Test AUC:", test_auc)
    print("Test Accuracy:", test_accuracy)

    # Make predictions on the test set
    predictions = model.predict(test_generator)
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

#================================= MAIN =================================

async def main():
    #generators = preprocessing(32)
    #model = create_model()
    #train_model(model, generators)
    #ml_results = evaluate_model(model, generators)
    grid_results = grid_search()

    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results)


if __name__ == "__main__":
    asyncio.run(main())


