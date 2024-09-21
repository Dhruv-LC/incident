import cv2
import numpy as np
import os
from dotenv import load_dotenv
from inference import get_model
from flask import Flask, Response
from datetime import datetime
import mysql.connector
import logging
import inference
from threading import Thread,Lock
from queue import Queue
from DroneBlocksTelloSimulator.DroneBlocksSimulatorContextManager import DroneBlocksSimulatorContextManager
from mysql.connector import pooling
import queue



dbconfig = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "atcc"
}

pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    **dbconfig
)

def get_connection():
    return pool.get_connection()

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY2")
if not api_key:
    raise ValueError("API key is not set. Please set the ROBOFLOW_API_KEY environment variable.")

# Initialize the Roboflow model
model = get_model("vids-jzo4g/7", api_key=api_key)
model2 = inference.get_model("fog-eiei5/2", api_key=api_key)

sim_key = "cd841ef8-51d7-420d-94a1-5bab389e6ae6"
drone_control_running = False

drone_control_lock = Lock()
# Define global variables
video_writer = None
video_filename = None
video_lock = Lock()



app = Flask(__name__)

# Buffer for vehicle positions and timestamps
vehicle_positions = {}
vehicle_timestamps = {}
vehicle_frames = {}  # Dictionary to store frames for each vehicle
stationary_duration = 5  # seconds
# Global dictionary to store the last timestamp of recorded incidents
last_recorded_incidents = {}

# Duration to prevent duplicate incidents (in seconds)
incident_record_duration = 10  # Adjust as needed


# Directory for saving incident images and videos
incident_images_dir = 'incident_images'
incident_videos_dir = 'incident_videos'
if not os.path.exists(incident_images_dir):
    os.makedirs(incident_images_dir)
if not os.path.exists(incident_videos_dir):
    os.makedirs(incident_videos_dir)


# Queue for frame processing
frame_queue = queue.Queue(maxsize=10)


def control_drone(sim_key):
    global drone_control_running
    with drone_control_lock:
        if drone_control_running:
            print("Drone control is already running.")
            return

        drone_control_running = True

    try:
        with DroneBlocksSimulatorContextManager(simulator_key=sim_key) as drone:
            drone.takeoff()
            drone.fly_forward(20, 'in')
            drone.fly_backward(20, 'in')
            drone.fly_left(20, 'in')
            drone.fly_right(20, 'in')
            drone.fly_up(20, 'in')
            drone.fly_down(20, 'in')
            drone.fly_to_xyz(10, 20, 30, 'in')
            drone.fly_curve(25, 25, 0, 0, 50, 0, 'in')
            drone.flip_forward()
            drone.flip_backward()
            drone.flip_left()
            drone.flip_right()
            drone.land()
    finally:
        with drone_control_lock:
            drone_control_running = False


def insert_incident(incident_type, image_name, video_name, ch_no):
    """Handle incident data (e.g., logging to database)."""
    incident_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    id = "Unknown"
    dir = "Unknown"
    connection = None
    cursor = None

    try:
        connection = get_connection()  # Use the pooled connection
        cursor = connection.cursor()
        sql = "INSERT INTO vids_event (ID, CurDateTime, event, dir, name, CH_NO, image) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (id, incident_datetime, incident_type, dir, video_name, ch_no, image_name)
        cursor.execute(sql, values)
        connection.commit()
        logging.info(f"Incident logged: {incident_type} at {incident_datetime}")
    except mysql.connector.Error as err:
        logging.error(f"Error inserting into VIDS: {err}")
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

        
def insert_VMS(incident_type, ch_no):
    """Handle VMS data (e.g., logging to database)."""
    incident_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "1"  # or whatever the status should be

    connection = None
    cursor = None

    try:
        # Get a connection from the pool
        connection = get_connection()  # Use the pooled connection
        cursor = connection.cursor()
        
        # Define SQL and values
        sql = "INSERT INTO vms_alert (location, message, created_date, isActive) VALUES (%s, %s, %s, %s)"
        values = (ch_no, incident_type, incident_datetime, status)
        
        # Execute the SQL command
        cursor.execute(sql, values)
        connection.commit()  # Commit the transaction

        # Log success
        logging.info(f"VMS alert logged: {incident_type} at {incident_datetime}")

    except mysql.connector.Error as err:
        # Log and handle errors
        logging.error(f"Error inserting into VMS table: {err}")
        if connection:
            connection.rollback()  # Rollback in case of error
    
    finally:
        # Ensure cursor and connection are closed properly
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        

def save_frame(frame, prefix):
    """Save the frame to disk with a timestamped filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f"{prefix}_{timestamp}.jpg"
    image_filename = os.path.join(incident_images_dir, image_name)
    cv2.imwrite(image_filename, frame)
    print(f"Saved image: {image_filename}")
    return image_name


def start_video_recording(incident_type, frames):
    """Start a new video recording."""
    global video_writer, video_filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = f"{incident_type}_{timestamp}.mp4"
    video_filename = os.path.join(incident_videos_dir, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))


def stop_video_recording():
    """Stop the current video recording."""
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"Saved video: {video_filename}")


def record_video(frames, ch_no, incident_type):
    """Record a video clip from frames and save it."""
    with video_lock:
        if video_writer is None:
            start_video_recording(incident_type, frames)
        
        for i, frame in enumerate(frames):
            if i >= 100:
                break
            video_writer.write(frame)
        
    return video_filename


def detect_stationary_vehicles(frame, class_name, confidence, ch_no, x1, y1, x2, y2):
    """Detect and handle stationary vehicles."""
    if class_name in ['Accident', 'NotAccident']:  # Check confidence level
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        vehicle_id = f"{cx}_{cy}"

        if vehicle_id in vehicle_positions:
            prev_x, prev_y = vehicle_positions[vehicle_id]
            if abs(prev_x - cx) < 10 and abs(prev_y - cy) < 10:
                if vehicle_id not in vehicle_timestamps:
                    vehicle_timestamps[vehicle_id] = datetime.now()
                    vehicle_frames[vehicle_id] = [frame]  # Start collecting frames
                else:
                    elapsed_time = (datetime.now() - vehicle_timestamps[vehicle_id]).total_seconds()
                    if elapsed_time > stationary_duration:
                        vehicle_frames[vehicle_id].append(frame)
                        if elapsed_time >= stationary_duration:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Stationary Vehicle',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            video_name = record_video(vehicle_frames[vehicle_id], ch_no, "stationary_vehicle")
                            insert_VMS("Stationary_Vehicle",ch_no)

                            insert_incident("Stationary_Vehicle", save_frame(frame, 'stationary_vehicle'), video_name,
                                            ch_no)
                            vehicle_timestamps[vehicle_id] = datetime.now()  # Reset timer
                            vehicle_frames[vehicle_id] = [frame]  # Restart frame collection
            else:
                vehicle_positions[vehicle_id] = (cx, cy)
                if vehicle_id in vehicle_timestamps:
                    del vehicle_timestamps[vehicle_id]
                    if vehicle_id in vehicle_frames:
                        del vehicle_frames[vehicle_id]
        else:
            vehicle_positions[vehicle_id] = (cx, cy)
            vehicle_frames[vehicle_id] = [frame]  # Start collecting frames for new vehicle


# def detect_objects(frame, ch_no):
#     """Detect objects in the frame using the Roboflow model."""
#     try:
#         img_array = np.asarray(frame)
#         _, img_encoded = cv2.imencode('.jpg', img_array)
#         img_bytes = img_encoded.tobytes()
#         results = model.infer(image=img_bytes)

#         if isinstance(results, list) and len(results) > 0:
#             inference_response = results[0]
#             if hasattr(inference_response, 'predictions'):
#                 predictions = inference_response.predictions
#                 for detection in predictions:
#                     if hasattr(detection, 'confidence') and hasattr(detection, 'class_name'):
#                         confidence = detection.confidence
#                         if confidence >= 0.1:  # Check confidence level
#                             class_name = detection.class_name
#                             x = int(detection.x)
#                             y = int(detection.y)
#                             width = int(detection.width)
#                             height = int(detection.height)
#                             x1 = int(x - width / 2)
#                             y1 = int(y - height / 2)
#                             x2 = int(x + width / 2)
#                             y2 = int(y + height / 2)

#                             # Draw bounding box and class name on the frame
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             cv2.putText(frame, f'{class_name} ({confidence * 100:.2f}%)',
#                                         (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

#                             # Detect and handle stationary vehicles
#                             detect_stationary_vehicles(frame, class_name, confidence, ch_no, x1, y1, x2, y2)

                         
#                             if class_name == 'Accident' and confidence >= 0.9:
#                                 image_name = save_frame(frame, 'accident')
#                                 video_name = record_video([frame], ch_no, "accident")
#                                 # Check if drone control is already running
#                                 # with drone_control_lock:
#                                 #      if not drone_control_running:
#                                 #          # Start the drone control in a separate thread
#                                 #          drone_thread = Thread(target=control_drone, args=(sim_key,))
#                                 #          drone_thread.start()
#                                 insert_VMS(class_name, ch_no)

#                                 insert_incident(class_name, image_name, video_name, ch_no)

                                

#                             # Save the frame if no helmet is detected
#                             if class_name == 'NoHelmet' and confidence >= 0.5:
#                                 image_name = save_frame(frame, 'no_helmet')
#                                 insert_incident(class_name, image_name, "NA", ch_no)

#                             # Save the frame if an animal is detected
#                             if class_name == 'Animal' and confidence >= 0.5:
#                                 image_name = save_frame(frame, 'Animal')
#                                 insert_incident(class_name, image_name, "NA", ch_no)

#                             # Save the frame if a pedestrian is detected
#                             if class_name == 'Pedestrian' and confidence >= 0.5:
#                                 image_name = save_frame(frame, 'Pedestrian')
#                                 insert_incident(class_name, image_name, "NA", ch_no)

#                             # Save the frame if triple riding is detected
#                             if class_name == 'Triple Riding' and confidence >= 0.5:
#                                 image_name = save_frame(frame, 'Triple_riding')
#                                 insert_incident(class_name, image_name, "NA", ch_no)

#                             # Save the frame if debris is detected
#                             if class_name == 'Debris' and confidence >= 0.5:
#                                 image_name = save_frame(frame, 'Debris')
#                                 insert_incident(class_name, image_name, "NA", ch_no)

#     except Exception as e:
#         print(f"Error during inference: {e}")

#     return frame

def detect_objects(frame, ch_no):
    """Detect objects in the frame using the Roboflow model."""
    try:
        img_array = np.asarray(frame)
        _, img_encoded = cv2.imencode('.jpg', img_array)
        img_bytes = img_encoded.tobytes()
        results = model.infer(image=img_bytes)

        if isinstance(results, list) and len(results) > 0:
            inference_response = results[0]
            if hasattr(inference_response, 'predictions'):
                predictions = inference_response.predictions
                for detection in predictions:
                    if hasattr(detection, 'confidence') and hasattr(detection, 'class_name'):
                        confidence = detection.confidence
                        if confidence >= 0.1:  # Check confidence level
                            class_name = detection.class_name
                            x = int(detection.x)
                            y = int(detection.y)
                            width = int(detection.width)
                            height = int(detection.height)
                            x1 = int(x - width / 2)
                            y1 = int(y - height / 2)
                            x2 = int(x + width / 2)
                            y2 = int(y + height / 2)

                            # Draw bounding box and class name on the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{class_name} ({confidence * 100:.2f}%)',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

                            # Check for stationary vehicles
                            detect_stationary_vehicles(frame, class_name, confidence, ch_no, x1, y1, x2, y2)

                            # Handle incidents
                            current_time = datetime.now()
                            last_time = last_recorded_incidents.get(class_name, None)

                            if class_name == 'Accident' and confidence >= 0.9:
                                with drone_control_lock:
                                     if not drone_control_running:
                                         # Start the drone control in a separate thread
                                         drone_thread = Thread(target=control_drone, args=(sim_key,))
                                         drone_thread.start()
                                if last_time is None or (current_time - last_time).total_seconds() > incident_record_duration:
                                    image_name = save_frame(frame, 'accident')
                                    video_name = record_video([frame], ch_no, "accident")
                                    last_recorded_incidents[class_name] = current_time  # Update last recorded time
                                    insert_VMS(class_name, ch_no)
                                    insert_incident(class_name, image_name, video_name, ch_no)

                            # Check for other incidents
                            incident_types = ['NoHelmet', 'Animal', 'Pedestrian', 'Triple Riding', 'Debris']
                            for incident in incident_types:
                                if class_name == incident and confidence >= 0.5:
                                    if last_recorded_incidents.get(incident, None) is None or (current_time - last_recorded_incidents[incident]).total_seconds() > incident_record_duration:
                                        image_name = save_frame(frame, incident.lower().replace(" ", "_"))
                                        insert_incident(incident, image_name, "NA", ch_no)
                                        last_recorded_incidents[incident] = current_time  # Update last recorded time

    except Exception as e:
        print(f"Error during inference: {e}")

    return frame



def detect_fog(frame):
    response = model2.infer(image=frame)

    # Print the response object to understand its structure
    # print(response)

    # Extract class and confidence from the response object
    try:
        if isinstance(response, list) and response:
            # Assuming response is a single object in a list
            first_result = response[0]

            # Use attributes directly
            class_label = first_result.top  # 'top' contains the top class name
            confidence = first_result.confidence  # 'confidence' contains the top confidence score
        else:
            class_label = 'Unknown'
            confidence = 0.0
    except AttributeError as e:
        print(f"Error accessing attributes: {e}")
        class_label = 'Unknown'
        confidence = 0.0

    # Prepare the text to be displayed
    text = f"Class: {class_label}, Confidence: {confidence:.2f}"

    # Set the font and position for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 50)
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2

    # Add text to the resized frame
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    return frame
    



def process_frame(frame, ch_no):
    """Process the frame using object detection and save incidents if needed."""
    frame = detect_fog(frame)
    frame = detect_objects(frame, ch_no)
    return frame


def generate_frames(video_path, ch_no):
    """Generate frames from the video file for streaming."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path} on Ch_No: {ch_no}.")
        return

    # original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_width = 600
    desired_height = 400

    frame_counter = 0  # To process every N-th frame
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Increment frame counter and skip processing if necessary
        frame_counter += 1
        if frame_counter % 2 != 0:  # Process every 5th frame
            continue

        # Resize frame while maintaining aspect ratio
        resized_frame = cv2.resize(frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

        # Add the frame to the queue for processing
        frame_queue.put((resized_frame, ch_no))

        # Process frames from the queue in separate threads
        if not frame_queue.empty():
            frame, ch_no = frame_queue.get()
            processing_thread = Thread(target=process_frame, args=(frame, ch_no))
            processing_thread.start()
            processing_thread.join()  # Wait for the thread to finish

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        if not ret:
            continue

        # Yield frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    stop_video_recording()


@app.route('/stream1')
def video_feed1():
    """Video streaming route for the first stream."""
    video_path = 'animal.mp4'  # Replace with your first video file path
    ch_no = "101"
    return Response(generate_frames(video_path, ch_no), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream2')
def video_feed2():
    """Video streaming route for the second stream."""
    video_path = 'fog.mp4'  # Replace with your second video file path
    ch_no = "102"
    return Response(generate_frames(video_path, ch_no), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream3')
def video_feed3():
    """Video streaming route for the second stream."""
    video_path = 'highway_accident.mp4'  # Replace with your second video file path
    ch_no = "103"
    return Response(generate_frames(video_path, ch_no), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream4')
def video_feed4():
    """Video streaming route for the second stream."""
    video_path = 'animal.mp4'  # Replace with your second video file path
    ch_no = "104"
    return Response(generate_frames(video_path, ch_no), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream5')
def video_feed5():
    """Video streaming route for the second stream."""
    video_path = 'without_helmet.mp4'  # Replace with your second video file path
    ch_no = "105"
    return Response(generate_frames(video_path, ch_no), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__== "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)