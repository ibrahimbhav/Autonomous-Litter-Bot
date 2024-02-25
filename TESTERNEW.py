# Litter detection functions
import cv2
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import pandas as pd
from picamera2 import Picamera2
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_vertices_with_circles(vertices, save_as=None):
    x_coordinates, y_coordinates = zip(*vertices)
    
    plt.plot(x_coordinates + (x_coordinates[0],), y_coordinates + (y_coordinates[0],), marker='o')
    plt.scatter(x_coordinates, y_coordinates, color='red', s=50)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('LITTER MAP')

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()

#plot_vertices_with_circles(vertices, save_as=save_filename)

# Initialize Picamera
picam2 = Picamera2()
picam2.start()

# Load the object detection model
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Set the directory to save snapshots
save_directory = "./snapshots/"
os.makedirs(save_directory, exist_ok=True)

width = 1080
height = 690

def capture_and_save_snapshot():
    # Capture frame from Picamera
    image = picam2.capture_array()
    image = cv2.flip(image, -1)

    # Resize to respect the input_shape
    inp = cv2.resize(image, (width, height))

    # Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Convert image to tensor
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    # Object detection
    boxes, scores, classes, num_detections = detector(rgb_tensor)

    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]

    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # Save the image with a unique name based on the number of snapshots taken
    save_path = os.path.join(save_directory, f"snapshot_{len(os.listdir(save_directory)) + 1}.png")
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Display the resulting frame
    img_boxes = image.copy()
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        score_txt = f'{100 * round(score, 0)}'
        img_boxes = cv2.rectangle(img_boxes, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.imshow('Snapshot', img_boxes)

    # Return detected object labels
    return set(pred_labels)

# Autonomous movement and mapping functions
from src import vehicle as vehicle_module
from src import distance_sensor as distance_sensor_module
import time
import src   

""" def check_near(total_time):
        vehicle.drive_forward(1)
        start_time = time.time()
        while(time.time() < start_time + total_time):
            while(1):
                if (distance_sensor1.distance < 0.4 or distance_sensor2.distance < 0.4):
                    print("OBJECT NEAR --- REDIRECT")
                    if (distance_sensor1.distance < distance_sensor2.distance):
                        while (distance_sensor1.distance < 0.4):
                            vehicle.pivot_right(0.1)
                        vehicle.stop()
                        time.sleep(1)
                        vehicle.drive_forward(1)
                        break
                        # KILL MOTION/MOMENTUM FOR CONTROL
                    else:
                        while (distance_sensor2.distance < 0.4):
                            vehicle.pivot_left(0.1)
                        vehicle.stop()
                        time.sleep(1)
                        vehicle.drive_forward(1)
                        break """

def dir_after_right(curr_dir):
  if (curr_dir == 0):
    return 1
  if (curr_dir == -1):
    return 0
  if (curr_dir == 2):
    return -1
  if (curr_dir == 1):
    return 2
def dir_after_left(curr_dir):
  if (curr_dir == 0):
    return -1
  if (curr_dir == 1):
    return 0
  if (curr_dir == 2):
    return 1
  if (curr_dir == -1):
    return 2
def update_coords(coords, direction, distance, mul):
  if (direction == 0):
    return (coords[0], coords[1] + mul * distance)
  if (direction == 1):
    return (coords[0] + mul * distance, coords[1])
  if (direction == -1):
    return (coords[0] - mul*distance, coords[1])
  if (distance == 2):
    return (coords[0], coords[1] - mul*distance)

def turn_right():
    vehicle.pivot_right(1)
    time.sleep(0.7)
    vehicle.stop()
def turn_left():
    vehicle.pivot_left(1)
    time.sleep(0.7)
    vehicle.stop()
def check_near(total_time, start_move, coords, trash_location, direction):
    i = 0
    vehicle.drive_forward(1)
    start_time = time.time()
    while(time.time() < start_time + total_time):
        now_time = time.time()
        distance = 10 * abs(start_move - now_time)
        if i > 0:
          print("----------------------------------------")
          print("UPDATING COORDS!")
          print("OLD COORDS =", coords)
          coords = update_coords(coords, direction, distance, 1)
          print("NEW COORDS =", coords)
          print("----------------------------------------")
        i = i + 1
        while(1):
            if (distance_sensor1.distance < 0.4 or distance_sensor2.distance < 0.4):
                print("OBJECT NEAR --- REDIRECT")
                vehicle.stop()
                machine_prediction = capture_and_save_snapshot()
                if (len(machine_prediction) > 0):
                  #LITTER IS HERE
                  vehicle.stop()
                  print(machine_prediction)
                  time.sleep(2)
                  trash_location.append(coords)
                  print("UPDATED COORDINATES: ", trash_location)
                if (distance_sensor1.distance < distance_sensor2.distance):
                    while (distance_sensor1.distance < 0.4):
                        print("----------------------------------------")
                        print("TURNING RIGHT")
                        print("CURRENT DIRECTION = ", direction)
                        turn_right()
                        direction = dir_after_right(direction)
                        print("NEW DIRECTION = ", direction)
                        print("----------------------------------------")
                    vehicle.stop()
                    time.sleep(1)
                    start_move = time.time()
                    vehicle.drive_forward(1)
                    break
                    # KILL MOTION/MOMENTUM FOR CONTROL
                else:
                    while (distance_sensor2.distance < 0.4):
                        print("----------------------------------------")
                        print("TURNING LEFT")
                        print("CURRENT DIRECTION = ", direction)
                        turn_left()
                        direction = dir_after_left(direction)
                        print("NEW DIRECTION = ", direction)
                    vehicle.stop()
                    time.sleep(1)
                    start_move = time.time()
                    vehicle.drive_forward(1)
                    break
    
if __name__ == '__main__':

    distance_sensor1 = distance_sensor_module.DistanceSensor({
            "pins": {
            "echo": 23,
            "trigger": 24
            }
            })

    distance_sensor2 = distance_sensor_module.DistanceSensor({
            "pins": {
            "echo": 17,
            "trigger": 27
            }
            })
        
    vehicle = vehicle_module.Vehicle(
        {
            "motors": {
                "left": {
                    "pins": {
                        "speed": 12,
                        "control1": 5,
                        "control2": 6
                    }
                },
                "right": {
                    "pins": {
                        "speed": 13,
                        "control1": 7,
                        "control2": 8
                    }
                }
            }
        }
    )    
    
#def check_near(total_time, start_move, coords, trash_location, direction):
    
    coords = (0.0, 0.0)
    #     0
    # -1     1
    #     2
    direction = 0
    rotation_speed = 0
    start_move = time.time()
    i = 0
    trash_location = []
    print("hi")
    check_near(10, start_move, coords, trash_location, direction)
    #check_near(10)
    vehicle.stop()
    plot_vertices_with_circles(trash_location,"test.png")