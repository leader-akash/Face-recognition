import os
import cv2
import numpy as np
from PIL import Image

def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech):
    # Create LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(haarcasecade_path)

    # Initialize lists for faces and IDs
    faces = []
    Ids = []

    # Function to get images and labels
    def getImagesAndLabels(directory):
        imagePaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert("L")  # Convert image to grayscale
            imageNp = np.array(pilImage, "uint8")
            Id = int(os.path.split(imagePath)[1].split("_")[0])  # Extract ID from file name
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids

    try:
        # Get faces and IDs from training image directory
        faces, Ids = getImagesAndLabels(trainimage_path)

        # Train recognizer with faces and IDs
        recognizer.train(faces, np.array(Ids))

        # Save trained model
        recognizer.save(trainimagelabel_path)

        # Display success message
        message.config(text="Training completed successfully")
        text_to_speech("Training completed successfully")
    except Exception as e:
        # Display error message
        error_message = f"Error during training: {str(e)}"
        message.config(text=error_message)
        text_to_speech(error_message)
