import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, jsonify, Response
import threading
import time

# Load the TensorFlow model and access the serving function
model = tf.saved_model.load("./ssd_mobilenet_v2_coco_2018_03_29/saved_model")
detect_fn = model.signatures['serving_default']

# COCO class labels
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "TV", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush"
}

# Initialize Flask app
app = Flask(__name__)

# Create a dictionary to track car presence in multiple boxes
grid_cars = {1: False, 2: False, 3: False, 4: False, 5: False}
previous_grid_cars = {1: False, 2: False, 3: False, 4: False, 5: False}  # Track previous state for exit detection

# Function to draw the First box
def draw_first_box(frame):
    height, width, _ = frame.shape
    # Define the dimensions for the first box
    box_width, box_height = 50, 100
    top_left = (width // 4 - box_width // 2 - 200, height // 4 - box_height // 2)
    bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(frame, "First Box", (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Function to draw the second box
def draw_second_box(frame):
    height, width, _ = frame.shape
    # Define dimensions and position for the second box
    box_width, box_height = 50, 100
    top_left = (width // 4 - box_width // 2 + 70, height // 4 - box_height // 2)
    bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(frame, "Second Box", (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Function to draw the third box further to the left
def draw_third_box(frame):
    height, width, _ = frame.shape
    # Define dimensions and position for the third box
    box_width, box_height = 50, 100
    third_top_left = (width // 2 - box_width // 2 - 30, height // 4 - box_height // 2)
    third_bottom_right = (third_top_left[0] + box_width, third_top_left[1] + box_height)
    
    # Draw the box with a more visible color and size
    cv2.rectangle(frame, third_top_left, third_bottom_right, (0, 0, 255), 2)  # Red box for visibility
    cv2.putText(frame, "Third Box", (third_top_left[0], third_top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Function to draw the fourth box further to the left
def draw_fourth_box(frame):
    height, width, _ = frame.shape
    # Define dimensions and position for the third box
    box_width, box_height = 50, 100
    fourth_top_left = (width // 1 - box_width // 2 - 650, height // 4 - box_height // 2)
    fourth_bottom_right = (fourth_top_left[0] + box_width, fourth_top_left[1] + box_height)
    
    # Draw the box with a more visible color and size
    cv2.rectangle(frame, fourth_top_left, fourth_bottom_right, (0, 0, 255), 2)  # Red box for visibility
    cv2.putText(frame, "Fourth Box", (fourth_top_left[0], fourth_top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Function to draw the fifth box further to the left
def draw_fifth_box(frame):
    height, width, _ = frame.shape
    # Define dimensions and position for the third box
    box_width, box_height = 50, 100
    fifth_top_left = (width // 1 - box_width // 2 - 350, height // 4 - box_height // 2)
    fifth_bottom_right = (fifth_top_left[0] + box_width, fifth_top_left[1] + box_height)
    
    # Draw the box with a more visible color and size
    cv2.rectangle(frame, fifth_top_left, fifth_bottom_right, (0, 0, 255), 2)  # Red box for visibility
    cv2.putText(frame, "Fifth Box", (fifth_top_left[0], fifth_top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Function to run detection on a single frame
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    bboxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    classes = detections['detection_classes'][0].numpy().astype(int)  # Classes
    scores = detections['detection_scores'][0].numpy()  # Confidence scores

    # Filter out weak detections
    detection_threshold = 0.1
    height, width, _ = frame.shape
    detected_objects = []
    for i in range(len(scores)):
        if scores[i] > detection_threshold:
            y_min, x_min, y_max, x_max = bboxes[i]
            # Convert to pixel coordinates
            (x_min, x_max, y_min, y_max) = (int(x_min * width), int(x_max * width), 
                                             int(y_min * height), int(y_max * height))

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Get the object label
            label = COCO_LABELS.get(classes[i], f"Object {classes[i]}")
            label_text = f"{label}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label_text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_objects.append((label, (y_min, x_min, y_max, x_max)))

    return frame, detected_objects

# Flask route to render the status of the grid squares
@app.route('/status')
def status():
    return jsonify(grid_cars)

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)

    # Set the resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw both boxes and run detection
            draw_first_box(frame)
            draw_second_box(frame)
            draw_third_box(frame)
            draw_fourth_box(frame)
            draw_fifth_box(frame)
            frame, detected_objects = detect_objects(frame)
            
            # Encode frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prototype')
def prototype():
    return render_template('prototype.html')  # Make sure you have this template


# Check if any part of the car's bounding box overlaps with the box
def is_partially_inside(x_min, x_max, y_min, y_max, box_top_left, box_bottom_right):
    box_x_min, box_y_min = box_top_left
    box_x_max, box_y_max = box_bottom_right

    # Check for overlap: horizontal and vertical intersection
    overlap_x = not (x_max < box_x_min or x_min > box_x_max)
    overlap_y = not (y_max < box_y_min or y_min > box_y_max)

    return overlap_x and overlap_y


# Update the run_detection function to check for the third box
def run_detection():
    global grid_cars, previous_grid_cars
    cap = cv2.VideoCapture(0)

    # Set the resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Draw all three boxes on the frame
        draw_first_box(frame)
        draw_second_box(frame)
        draw_third_box(frame)
        draw_fourth_box(frame)
        draw_fifth_box(frame)

        # Run object detection
        frame, detected_objects = detect_objects(frame)

        # Define the boundaries for each box
        height, width, _ = frame.shape
        first_box_width, first_box_height = 50, 100
        first_top_left = (width // 4 - first_box_width // 2 -200, height // 4 - first_box_height // 2)
        first_bottom_right = (first_top_left[0] + first_box_width, first_top_left[1] + first_box_height)

        second_box_width, second_box_height = 50, 100
        second_top_left = (width // 4 - second_box_width // 2 + 70, height // 4 - second_box_height // 2)
        second_bottom_right = (second_top_left[0] + second_box_width, second_top_left[1] + second_box_height)

        third_box_width, third_box_height = 50, 100
        third_top_left = (width // 2 - third_box_width // 2 - 30, height // 4 - third_box_height // 2)
        third_bottom_right = (third_top_left[0] + third_box_width, third_top_left[1] + third_box_height)

        fourth_box_width, fourth_box_height = 50, 100
        fourth_top_left = (width // 1 - fourth_box_width // 2 - 650, height // 4 - fourth_box_height // 2)
        fourth_bottom_right = (fourth_top_left[0] + fourth_box_width, fourth_top_left[1] + fourth_box_height)
        
        fifth_box_width, fifth_box_height = 50, 100
        fifth_top_left = (width // 1 - fifth_box_width // 2 - 350, height // 4 - fifth_box_height // 2)
        fifth_bottom_right = (fifth_top_left[0] + fifth_box_width, fifth_top_left[1] + fifth_box_height)

        # Update grid_cars status based on detections in each box
        cars_in_first_box = any(
            'car' in label.lower() and 
            is_partially_inside(x_min, x_max, y_min, y_max, first_top_left, first_bottom_right)
            for label, (y_min, x_min, y_max, x_max) in detected_objects)

        cars_in_second_box = any(
            'car' in label.lower() and 
            is_partially_inside(x_min, x_max, y_min, y_max, second_top_left, second_bottom_right)
            for label, (y_min, x_min, y_max, x_max) in detected_objects)

        cars_in_third_box = any(
            'car' in label.lower() and 
            is_partially_inside(x_min, x_max, y_min, y_max, third_top_left, third_bottom_right)
            for label, (y_min, x_min, y_max, x_max) in detected_objects)

        cars_in_fourth_box = any(
            'car' in label.lower() and 
            is_partially_inside(x_min, x_max, y_min, y_max, fourth_top_left, fourth_bottom_right)
            for label, (y_min, x_min, y_max, x_max) in detected_objects)
        
        cars_in_fifth_box = any(
            'car' in label.lower() and 
            is_partially_inside(x_min, x_max, y_min, y_max, fifth_top_left, fifth_bottom_right)
            for label, (y_min, x_min, y_max, x_max) in detected_objects)

        # Update the grid_car status for all three boxes
        grid_cars[1] = cars_in_first_box
        grid_cars[2] = cars_in_second_box
        grid_cars[3] = cars_in_third_box
        grid_cars[4] = cars_in_fourth_box
        grid_cars[5] = cars_in_fifth_box

        # Only change to 'exit' if there was a change from car detected to no car detected
        if grid_cars != previous_grid_cars:
            for key, value in grid_cars.items():
                if previous_grid_cars[key] == True and value == False:
                    print(f"Exit detected in box {key}")
            previous_grid_cars = grid_cars.copy()

        # Wait for a short time to simulate frame capture rate
        time.sleep(0.1)

    cap.release()


# Start the detection thread
detection_thread = threading.Thread(target=run_detection)
detection_thread.daemon = True
detection_thread.start()

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
