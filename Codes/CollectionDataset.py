import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import time
from picamera2 import Picamera2
import numpy as np
import os

# Function to resize an image while maintaining its aspect ratio
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

def preprocess_and_extract_roi(image_path, output_path):
    image = cv2.imread(image_path)  # Read the image from the specified path
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}.")  # Print an error message if the image is not found
        return None  # Return None if the image is not found

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

    cv2.imwrite(output_path, roi_mirror_horizontal)  # Save the 128x128 ROI to the specified output path
    print(f"ROI saved at {output_path}")  # Print confirmation of saved ROI
    return roi_mirror_horizontal  # Return the resized ROI


# Function to find valley points in a contour
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

# Function to calculate new points based on valley points
def calculate_new_points(valley_points):
    point1 = np.array(valley_points[0])  # Convert the first valley point to a NumPy array
    point2 = np.array(valley_points[1])  # Convert the second valley point to a NumPy array
    distance = np.linalg.norm(point2 - point1)  # Calculate the distance between the two points
    direction = (point2 - point1) / distance  # Calculate the direction vector between the points
    perpendicular_direction = np.array([-direction[1], direction[0]])  # Calculate the perpendicular direction

    new_point1 = point1 + perpendicular_direction * distance  # Calculate the new point 1 by moving perpendicular to the direction
    new_point2 = point2 + perpendicular_direction * distance  # Calculate the new point 2 by moving perpendicular to the direction

    return (new_point1.astype(int), new_point2.astype(int))  # Return the new points as integers

# Function to get the warped image (ROI) based on valley points
def get_warped_image(resized_image, valley_points):
    new_points = calculate_new_points(valley_points)  # Calculate new points based on valley points
    src_pts = np.array([valley_points[0], valley_points[1], new_points[1], new_points[0]], dtype="float32")  # Define source points for perspective transform
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")  # Define destination points for perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # Get the perspective transform matrix
    warped = cv2.warpPerspective(resized_image, M, (100, 100))  # Apply perspective transform to get the ROI
    return warped  # Return the warped image

# Function to display the results
def display_results(image, edges, binary_image, denoised_image, hog_image_rescaled, resized_image, valley_points):
    cv2.imshow("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the original image
    cv2.imshow("Edges", edges)  # Display the edges
    cv2.imshow("Binary Image", binary_image)  # Display the binary image
    cv2.imshow("Denoised Image", cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)) # Display the denoised image
    cv2.imshow("HOG Features", hog_image_rescaled)  # Display the HOG features

    new_points = calculate_new_points(valley_points)  # Calculate new points based on valley points

    cv2.imshow("Resized Image", cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # Display the resized image
    for point in valley_points:
        cv2.circle(resized_image, point, 5, (0 , 0, 255), -1)  # Plot valley points A and B
    for point in new_points:
        cv2.circle(resized_image, point, 5, (0, 255, 0), -1)  # Plot valley points C and D
    cv2.line(resized_image, valley_points[0], valley_points[1], (0, 255, 0), 2)  # Draw a green line connecting valley points A and B
    cv2.line(resized_image, valley_points[1], new_points[1], (0, 255, 0), 2)  # Draw a green line connecting valley point B and new point D
    cv2.line(resized_image, new_points[1], new_points[0], (0, 255, 0), 2)  # Draw a green line connecting new points D and C
    cv2.line(resized_image, new_points[0], valley_points[0], (0, 255, 0 ), 2)  # Draw a green line connecting new point C and valley point A
    cv2.imshow("Resized Image", cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # Display the resized image

    warped = get_warped_image(resized_image, valley_points)  # Get the warped image (ROI)
    cv2.imshow("Region of Interest", cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))  # Display the ROI
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Ask for the person's name for distinguishing the image sets
person_name = input("Please enter the person's name: ").strip()

# Ensure the name is valid for filenames
if not person_name.isalnum():
    raise ValueError("Person's name should be alphanumeric (letters and digits only).")

# Use MediaPipe to draw the hand framework over the top of hands it identifies in real-time
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# Set up Picamera2 for capturing video from the Noir camera module
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (2592, 1944)})
picam2.configure(camera_config)
picam2.start()

# Wait for 10 seconds before starting to capture images
start_time = time.time()
while time.time() - start_time < 10:
    # Display a message to inform the user about the wait time
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))
    cv2.putText(frame, "Please get ready. Starting in 10 seconds...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Calculate remaining time and display it
    remaining_time = 10 - int(time.time() - start_time)
    cv2.putText(frame, f"Starting in: {remaining_time}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame using OpenCV
    cv2.imshow("Frame", frame)  # Display the frame
    cv2.waitKey(1)  # Wait for a short period
# Clear the window for the main capture process to begin
cv2.destroyAllWindows()  # Close all OpenCV windows

# Directories for training, validation, and testing
dirs = ['TrainingSet', 'ValidationSet', 'TestingSet']

# Create directories if they don't exist
for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create directories for training and validation if they don't exist
train_dir = 'train1'
val_dir = 'val1'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Total number of images to be taken
total_images = 6

# Counters to keep track of images stored in each set
image_counter = 0

# Initialize the last captured time
last_captured_time = time.time()

# Define the capture interval (in seconds)
capture_interval = 2  # For example, capture every 2 seconds

# Add confidence values and extra settings to MediaPipe hand tracking
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while True:
        # Capture frame from the Pi camera
        frame = picam2.capture_array()

        # Check if the frame is in the correct format (ensure it's 3-channel BGR)
        if frame.shape[-1] == 4:  # If it 's 4 channels (like RGBA), convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[-1] == 3:  # If it's RGB, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize frame for consistent processing
        frame1 = cv2.resize(frame, (2592, 1944))

        # Process the frame to detect hands
        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        # Make a copy of the frame for saving, without text
        frame_to_save = frame1.copy()

        # Add a message to instruct the user to open their hand fully
        if image_counter < total_images:
            cv2.putText(frame1, "Please open your hand fully!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the displayed frame
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                # Show the current frame with the text using OpenCV
                cv2.imshow("Frame", frame1)  # Use OpenCV to show the frame
                key = cv2.waitKey(1)  # Wait for a short period

                # Get the current time
                current_time = time.time()

                # Check if the interval has passed since the last capture
                if current_time - last_captured_time > capture_interval and image_counter < total_images:
                    # Decide which directory the image will go to based on the image counter
                    if image_counter < 4:  # First 4 images go to train
                        image_dir = train_dir
                    else:  # Last 2 images go to val
                        image_dir = val_dir
                    img_dir = 'images'
                    os.makedirs(img_dir, exist_ok=True)
                    folder_path_roi=f'{image_dir}/{person_name}'
                    folder_path_image=f'images/{person_name}'
                  
                    os.makedirs(folder_path_roi, exist_ok=True)
                    os.makedirs(folder_path_image, exist_ok=True)
                    # Save the frame to the selected directory with the person's name
                    image_path = f'images/{person_name}/{person_name}image{image_counter}.bmp'
                    output_path = f'{image_dir}/{person_name}/{person_name}-{image_counter}.bmp'

                    cv2.imwrite(image_path, frame_to_save)
                    print(f"Image saved at {image_path}")

                    # Show message that the image was saved successfully
                    cv2.putText(frame1, f"Image {image_counter + 1} saved successfully!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame1, "Please put your hand again.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Display the success message for 2 seconds
                    cv2.imshow("Frame", frame1)
                    cv2.waitKey(2000)

                    # Process and save the ROI
                    roi = preprocess_and_extract_roi(image_path, output_path)


                    # Update the time and image counter
                    last_captured_time = current_time
                    image_counter += 1

                    # If all images have been taken, display the thank you message and exit
                    if image_counter >= total_images:
                        frame1 = cv2.resize(frame, (2592, 1944))
                        cv2.putText(frame1, "Thank you!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255, 0), 2)
                        cv2.putText(frame1, "Your hand images have been captured successfully.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Show the final thank you message for 3 seconds
                        cv2.imshow("Frame", frame1)
                        cv2.waitKey(3000)  # Display the final message for 3 seconds
                        picam2.close()
                        cv2.destroyAllWindows()
                        exit()

        # Press 'q' to exit the loop manually (in case needed)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# Release resources and close windows
picam2.close()
cv2.destroyAllWindows()