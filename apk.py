import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load the trained model
model = load_model('stroke_detection_model.h5')

# Load VGG16 for feature extraction
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

# Variables to hold predictions and true labels for evaluation
predictions = []
true_labels = []

# Function to extract features from an input image
def extract_features(img_path, feature_extractor):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# Function to predict stroke from an image
def predict_stroke(img_path):
    features = extract_features(img_path, feature_extractor)
    features = np.expand_dims(features, axis=0)  # Reshape for prediction
    prediction = model.predict(features)[0][0]
    
    if prediction > 0.5:
        return "Normal (Benign)", 1, prediction  # Return prediction probability
    else:
        return "Acute Ischemic Stroke (Malignant)", 0, prediction

# Function to load and display the image, perform the prediction, and show options for back or re-upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((640, 640))  # Resize the image for display (640x640)
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk  # To prevent garbage collection

            # Perform prediction
            result, prediction_label, prediction_prob = predict_stroke(file_path)
            
            # Append prediction for evaluation
            predictions.append(prediction_prob)  # Store probability for ROC-AUC
            true_labels.append(prediction_label)  # Store true label (0 or 1)

            # Display the prediction result
            messagebox.showinfo("Prediction Result", f"The image is predicted as: {result}")

            # Show options to go back or re-upload an image
            open_back_or_reupload_options()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the image. Error: {str(e)}")

# Function to calculate evaluation metrics

def evaluate_model():
    if len(true_labels) > 0:
        # Convert predictions to binary labels based on threshold
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

        # Confusion Matrix
        confusion = confusion_matrix(true_labels, binary_predictions)
        
        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

        # Classification Report
        report = classification_report(true_labels, binary_predictions, target_names=["Malignant", "Benign"], digits=2, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Plot classification report as a table
        plt.figure(figsize=(8, 4))
        plt.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc="center", cellLoc="center")
        plt.axis("off")
        plt.title("Classification Report")
        plt.show()

        # ROC-AUC Score
        roc_auc = roc_auc_score(true_labels, predictions)
        fpr, tpr, _ = roc_curve(true_labels, predictions)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    else:
        messagebox.showinfo("Evaluation Results", "No images have been evaluated yet.")

# Function to reset the model for re-uploading
def reset_model():
    global predictions, true_labels
    predictions = []
    true_labels = []
    panel.config(image='')  # Clear the image display

# Function to open a dialog box with "Back" and "Re-upload" options
def open_back_or_reupload_options():
    option_window = tk.Toplevel(root)
    option_window.title("Options")
    option_window.geometry("300x150")

    # Button to go back (clears image display)
    back_btn = tk.Button(option_window, text="Back", command=lambda: clear_image(option_window), font=('Arial', 12))
    back_btn.pack(pady=10)

    # Button to re-upload a new image
    reupload_btn = tk.Button(option_window, text="Re-upload", command=lambda: reupload_image(option_window), font=('Arial', 12))
    reupload_btn.pack(pady=10)

# Function to clear image and close the options window
def clear_image(option_window):
    panel.config(image='')  # Clear the displayed image
    option_window.destroy()  # Close the options window

# Function to re-upload image without clearing predictions
def reupload_image(option_window):
    option_window.destroy()  # Close the options window
    upload_image()  # Trigger image upload again

# Set up the GUI
root = tk.Tk()
root.title("Stroke Detection from MRI/CT Scan")
root.geometry("700x700")  # Set the window size to 700x700

# Create a label for displaying the image
panel = tk.Label(root)
panel.pack(pady=20)  # Add some padding for better layout

# Create a button to upload the image
upload_btn = tk.Button(root, text="Upload MRI/CT Scan", command=upload_image, font=('Arial', 14))
upload_btn.pack(pady=10)  # Padding between button and image

# Create a button to evaluate the model
evaluate_btn = tk.Button(root, text="Evaluate Model", command=evaluate_model, font=('Arial', 14))
evaluate_btn.pack(pady=10)

# Create a button to reset the model for re-uploading images
reset_btn = tk.Button(root, text="Reset Model", command=reset_model, font=('Arial', 14))
reset_btn.pack(pady=10)

# Start the GUI loop
root.mainloop()
