import pydicom
import numpy as np
from PIL import Image
import os
import time

def dicom_to_image(dicom_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(dicom_folder):
        for filename in files:
            if filename.endswith('.dcm'):
                dicom_path = os.path.join(root, filename)
                dicom_data = pydicom.dcmread(dicom_path)
                img_array = dicom_data.pixel_array
                
                # Normalize image data
                img_array = img_array / np.max(img_array)
                
                # Convert array to image
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                
                # Generate a unique suffix based on timestamp
                timestamp = int(time.time() * 1000)  # Current time in milliseconds
                unique_suffix = f"_{timestamp}"
                
                # Construct the output filename with the unique suffix
                base_name = os.path.splitext(filename)[0]
                output_filename = base_name + unique_suffix + '.png'
                output_path = os.path.join(output_folder, os.path.relpath(dicom_path, dicom_folder))
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the image
                img.save(output_path.replace('.dcm', f'_{timestamp}.png'))

if __name__ == "__main__":
    dicom_folder = 'data'  # Base directory containing subdirectories
    output_folder = 'converted_images'
    dicom_to_image(dicom_folder, output_folder)
