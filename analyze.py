import numpy as np
import os

def analyze_clusters(labels_file, images_folder):
    # Load labels and file names
    labels = np.load(labels_file)
    file_names = np.load('file_names.npy')
    
    # Verify the number of labels and file names
    print(f"Number of labels: {len(labels)}")
    print(f"Number of file names: {len(file_names)}")
    
    # Check if the lengths match
    if len(labels) != len(file_names):
        raise ValueError("Mismatch between number of labels and file names.")
    
    # Gather all PNG files from subdirectories
    image_files = []
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith('.png'):
                relative_path = os.path.relpath(os.path.join(root, file), images_folder)
                image_files.append(relative_path)
    
    # Verify the number of image files
    print(f"Number of image files: {len(image_files)}")
    
    # Ensure that the length of image_files matches the number of labels
    if len(image_files) != len(labels):
        raise ValueError("Mismatch between number of image files and number of labels.")
    
    # Print label assignments
    for i, label in enumerate(labels):
        print(f"Image: {file_names[i]} - Label: {label}")

if __name__ == "__main__":
    # Now point to the CT folder with benign and malignant images
    analyze_clusters('labels.npy', 'CT')
