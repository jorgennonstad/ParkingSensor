# Parking Space Detection System

This project utilizes a live camera feed and object detection to monitor parking spaces. It uses TensorFlow's pre-trained SSD MobileNet v2 model for real-time object detection to identify cars in defined parking spots. The system then sends the status of these parking spaces (whether they are occupied or not) to ThingSpeak for remote monitoring.

## Key Features:
- **Real-Time Parking Monitoring**: Detects vehicles in five designated parking spots using live camera footage.
- **ThingSpeak Integration**: Sends real-time parking status to ThingSpeak to track availability.
- **API Calls**: Provides an API for external systems to retrieve the parking status.
- **Web Interface**: Displays the current parking status on a website with live updates.

## Technologies Used:
- **Backend**: Python, Flask, OpenCV, TensorFlow
- **Frontend**: HTML, JavaScript, CSS
- **Data Visualization**: ThingSpeak (IoT platform)

## How it Works:
- **Detection**: A webcam captures the video feed, and the system detects cars using TensorFlow's SSD model.
- **Parking Spots**: The camera feed is divided into 5 predefined parking spots, and the system tracks car occupancy in each.
- **Data Storage**: The parking status (occupied or free) is sent to ThingSpeak via its API.
- **Live Updates**: A website fetches real-time data from ThingSpeak and displays the current parking space status.

## Installation:
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
