import os
import time
from picamera2 import Picamera2
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
import numpy as np


# Use Agg backend for matplotlib (non-interactive)
import matplotlib
matplotlib.use('Agg')

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_new_points(valley_points):
    point1 = np.array(valley_points[0])  # Convert the first valley point to a NumPy array
    point2 = np.array(valley_points[1])  # Convert the second valley point to a NumPy array
    distance = np.linalg.norm(point2 - point1)  # Calculate the distance between the two points
    direction = (point2 - point1) / distance  # Calculate the direction vector between the points
    perpendicular_direction = np.array([-direction[1], direction[0]])  # Calculate the perpendicular direction

    new_point1 = point1 + perpendicular_direction * distance  # Calculate the new point 1 by moving perpendicular to the direction
    new_point2 = point2 + perpendicular_direction * distance  # Calculate the new point 2 by moving perpendicular to the direction

    return (new_point1.astype(int), new_point2.astype(int))  # Return the new points as integers

def find_valley_points(contour):
    hull = cv2.convexHull(contour, returnPoints=False)  # Compute the convex hull of the contour
    defects = cv2.convexityDefects(contour, hull)  # Find convexity defects in the contour
    valley_points = []  # Initialize a list to store valley points

    if defects is not None:  # Check if defects are found
        contour_area = cv2.contourArea(contour)
        dynamic_threshold = contour_area * 0.15  # Adjust the multiplier based on testing

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far = tuple(contour[f][0])  # Get the farthest point of the defect
            if d > dynamic_threshold:  # Check if the defect depth is greater than the threshold (adjust if necessary)
                valley_points.append(far)  # Append the farthest point to the valley points list

    # Ensure valley points are ordered by x-coordinate and are distinct
    valley_points = sorted(valley_points, key=lambda x: x[0])

    # Adjust the position of each valley point
    valley_points[0] = (valley_points[0][0] - 10, valley_points[0][1])
    valley_points[2] = (valley_points[2][0] + 8, valley_points[2][1])

    return [valley_points[0], valley_points[2]]  # Return the adjusted valley points

# Load pre-trained model
def load_model(model_path, num_classes):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 159)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Preprocessing function for prediction
def preprocess_for_model(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat grayscale to 3 channels
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image  # Return original image if no width or height is provided
    h, w = image.shape[:2]  # Get the height and width of the image
    if width is None:
        new_height = int((height / h) * w)  # Calculate new width to maintain aspect ratio
        resized_image = cv2.resize(image, (new_height, height))  # Resize image to new height
    else:
        new_width = int((width / w) * h)  # Calculate new height to maintain aspect ratio
        resized_image = cv2.resize(image, (width, new_width))  # Resize image to new width
    return resized_image  # Return the resized image

# Function to get the warped image (ROI) based on valley points
def get_warped_image(resized_image, valley_points):
    new_points = calculate_new_points(valley_points)  # Calculate new points based on valley points
    src_pts = np.array([valley_points[0], valley_points[1], new_points[1], new_points[0]], dtype="float32")  # Define source points for perspective transform
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")  # Define destination points for perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # Get the perspective transform matrix
    warped = cv2.warpPerspective(resized_image, M, (100, 100))  # Apply perspective transform to get the ROI
    return warped  # Return the warped image
# Function to extract ROI from an image
def extract_roi(image):
    resized_image = resize_image(image, width=800)  # Resize the image to a width of 800 pixels
    denoised_image = cv2.fastNlMeansDenoisingColored(resized_image, None, 11, 6, 7, 21)  # Denoise the image
    edges = cv2.Canny(denoised_image, 100, 200)  # Detect edges using Canny edge detector
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold the image to binary using Otsu's method
    kernel = np.ones((5, 5), np.uint8)  # Create a kernel for morphological operations
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Perform closing morphological transformation
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)  # Perform opening morphological transformation
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary image

    if not contours:
        print("No contours found!")  # Print an error message if no contours are found
        return None  # Exit the function

    largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour by area
    valley_points = find_valley_points(largest_contour)  # Find valley points in the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)  # Get the bounding rectangle of the largest contour
    roi = resized_image[y:y + h, x:x + w]  # Extract the region of interest (ROI) from the resized image
    
    if roi.size == 0:
        print("Error: ROI is empty!")  # Print an error message if the ROI is empty
        return None  # Return None if the ROI is empty
    else:
        print(f"ROI extracted with shape: {roi.shape}")  # Print the shape of the ROI

    warped = get_warped_image(resized_image, valley_points)  # Get the warped image (ROI)
    roi_100x100 = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    roi_128x128 = cv2.resize(roi_100x100, (128, 128))  # Resize the warped image to 128x128 pixels

    roi_mirror_horizontal = cv2.flip(roi_128x128, 1)
    return Image.fromarray(roi_mirror_horizontal)


# Predict the class of the ROI
def predict_image(roi_image, model, class_names, authorized_classes=None, unknown_threshold=0):
    if authorized_classes is None:
        authorized_classes = class_names
    
    image_tensor = preprocess_for_model(roi_image)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        confidence = confidence.item()
        
        if confidence < unknown_threshold:
            predicted_class = "unknown"
            authorized = False
            GPIO.output(11,1)
            time.sleep(1)
            GPIO.output(11,0)
            time.sleep(1)
            GPIO.output(11,1)
      
        else:
            predicted_class = class_names[predicted_idx.item()]
            authorized = predicted_class in authorized_classes
            GPIO.output(11,1)
            time.sleep(1)
            GPIO.output(11,0)
            time.sleep(1)
            GPIO.output(11,1)
    
    print(f"Predicted Class: {predicted_class} | Authorized: {'Yes' if authorized else 'No'} | Confidence: {confidence:.2f}")
    return predicted_class, authorized, confidence

# Display image using matplotlib
def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to capture an image from the Raspberry Pi camera
def capture_image():
    print("Capturing image from camera...")
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"size": (2592, 1944)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Wait for the camera to adjust
    frame = picam2.capture_array()
    picam2.stop()
    picam2.close()
    return frame

# Function to get class names dynamically from a folder
def get_class_names(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")
    class_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    return sorted(class_names)  # Sort for consistency

# Main function to integrate ROI extraction and prediction
def main(data_dir, model_path, authorized_classes=None, unknown_threshold=0.1):
    # Get class names dynamically
    print("Reading class names from folder...")
    class_names = get_class_names(os.path.join(data_dir, 'train'))
    print(f"Class Names: {class_names}")

    print("Capturing image...")
    frame = capture_image()

    # Optionally save the captured image
    captured_image_path = "captured_image.jpg"
    cv2.imwrite(captured_image_path, frame)
    print(f"Captured image saved at {captured_image_path}")

    print("Extracting ROI...")
    roi_image = extract_roi(frame)

    # Optionally save the ROI image
    roi_image_path = "roi_image.jpg"
    roi_image.save(roi_image_path)
    print(f"ROI image saved at {roi_image_path}")

    print("Loading model...")
    model = load_model(model_path, len(class_names))

    print("Predicting class...")
    predict_image(roi_image, model, class_names, authorized_classes, unknown_threshold)

# Example usage
if __name__ == "__main__":
    data_dir = "/home/raspberrypi5/Desktop/Our DataSet"  # Replace with the path to your dataset
    model_path = "/home/raspberrypi5/Downloads/redy159ourDataSet.pt"
    unknown_threshold = 0.1
    
    # Adjust as needed

    main(data_dir, model_path, unknown_threshold=unknown_threshold)