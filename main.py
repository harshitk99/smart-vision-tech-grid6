import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model and label encoder
model = load_model('quality_control_model.h5')  # Example CNN model for classification
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')  # Saved classes for product classification

# 1. Image Acquisition
def acquire_image(camera_port=0):
    cap = cv2.VideoCapture(camera_port)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Captured Image', frame)
        cv2.imwrite('captured_product.jpg', frame)
        cap.release()
        cv2.destroyAllWindows()
        return frame
    else:
        print("Failed to capture image")
        return None

# 2. Image Preprocessing
def preprocess_image(image):
    # Convert to grayscale for consistency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize image
    norm_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Apply Gaussian blur for noise removal
    blurred = cv2.GaussianBlur(norm_image, (5, 5), 0)
    return blurred

# 3. Feature Extraction (e.g., contour and edge detection)
def extract_features(image):
    # Detect edges using Canny
    edges = cv2.Canny(image, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 4. Object Detection and Classification
def classify_product(image):
    # Resize image to the size expected by the model
    resized_image = cv2.resize(image, (128, 128))  # Resize to 128x128 for CNN
    reshaped_image = resized_image.reshape(1, 128, 128, 1) / 255.0  # Normalize

    # Predict using the model
    predictions = model.predict(reshaped_image)
    predicted_class = np.argmax(predictions, axis=1)
    product_label = label_encoder.inverse_transform(predicted_class)
    
    return product_label[0]

# 5. Output and Feedback
def output_results(product_label, contours):
    print(f"Identified Product: {product_label}")
    print(f"Detected {len(contours)} potential defects or features.")

# Main function to integrate all components
def quality_control_system():
    # Step 1: Acquire the image from the camera
    image = acquire_image()
    if image is None:
        return

    # Step 2: Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Step 3: Extract features like contours and edges
    contours = extract_features(preprocessed_image)

    # Step 4: Classify the product using a trained model
    product_label = classify_product(preprocessed_image)

    # Step 5: Output and provide feedback on product quality
    output_results(product_label, contours)

if __name__ == "__main__":
    quality_control_system()
