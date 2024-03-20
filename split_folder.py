import os
import shutil
import random

def split_data(source, training, validation, split_ratio):
    for class_name in os.listdir(source):
        class_source_folder = os.path.join(source, class_name)
        class_training_folder = os.path.join(training, class_name)
        class_validation_folder = os.path.join(validation, class_name)
        
        os.makedirs(class_training_folder, exist_ok=True)
        os.makedirs(class_validation_folder, exist_ok=True)
        
        files = os.listdir(class_source_folder)
        num_files = len(files)
        num_training = int(num_files * split_ratio)
        random.shuffle(files)
        
        training_files = files[:num_training]
        validation_files = files[num_training:]
        
        for file in training_files:
            shutil.copy(os.path.join(class_source_folder, file), os.path.join(class_training_folder, file))
            
        for file in validation_files:
            shutil.copy(os.path.join(class_source_folder, file), os.path.join(class_validation_folder, file))


# Example usage:
source_folder = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/train_and_val"
training_folder = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/train"
validation_folder = "/home/gerpapa/Wildfire-Classification-CNN/dataset/forest_fire/valid"
split_ratio = 0.8  # 80% training, 20% validation

split_data(source_folder, training_folder, validation_folder, split_ratio)
