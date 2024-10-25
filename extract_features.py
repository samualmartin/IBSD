import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Load a pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

def extract_features_from_folder(folder_path):
    features = []
    labels = []
    file_names = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.png'):
                img_path = os.path.join(root, filename)
                img_features = extract_features(img_path)
                features.append(img_features)
                
                # Label: benign = 1, malignant = 0
                if 'benign' in root:
                    labels.append(1)  # normal
                elif 'malignant' in root:
                    labels.append(0)  # ischemic stroke
                
                file_names.append(os.path.relpath(img_path, folder_path))
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    print("Extracted features shape:", features_array.shape)
    return features_array, labels_array, file_names

if __name__ == "__main__":
    folder_path = 'CT'  # Points to the folder with 'benign' and 'malignant' subfolders
    features, labels, file_names = extract_features_from_folder(folder_path)
    
    # Save features, labels, and file names
    np.save('features.npy', features)
    np.save('labels.npy', labels)  # Save labels directly from the benign/malignant mapping
    np.save('file_names.npy', np.array(file_names))
