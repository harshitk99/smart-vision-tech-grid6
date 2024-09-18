import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('quality_control_model.h5')  
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')  


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


def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm_image = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(norm_image, (5, 5), 0)
    return blurred

def extract_features(image):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_product(image):
    resized_image = cv2.resize(image, (128, 128))  
    reshaped_image = resized_image.reshape(1, 128, 128, 1) / 255.0  


    predictions = model.predict(reshaped_image)
    predicted_class = np.argmax(predictions, axis=1)
    product_label = label_encoder.inverse_transform(predicted_class)
    
    return product_label[0]

def output_results(product_label, contours):
    print(f"Identified Product: {product_label}")
    print(f"Detected {len(contours)} potential defects or features.")


def quality_control_system():

    image = acquire_image()
    if image is None:
        return


    preprocessed_image = preprocess_image(image)


    contours = extract_features(preprocessed_image)

    product_label = classify_product(preprocessed_image)

    output_results(product_label, contours)

if __name__ == "__main__":
    quality_control_system()
