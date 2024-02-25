from src import vehicle as vehicle_module
from src import distance_sensor as distance_sensor_module
import time   


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
                        break
"""
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
coords = (0, 0)
#     0
# -1     1
#     2
direction = 0
rotation_speed = 0
start_move = time.time()
i = 0
trash_location = []
''' def turn_left():
  #TODO: IMPLEMENT THIS
  return 0
def turn_right():
  #TODO: IMPLEMENT THIS
  return 0 '''
def check_near(time,prev_funct):
    prev_funct()
    start_time = time.time()
    while(time.time() < start_time + time):
        now_time = time.time()
        distance = abs(start_move - now_time)
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
                machine_prediction = capture_and_save_snapshot()
                if (len(machine_prediction) > 0):
                  #LITTER IS HERE
                  print(machine_prediction)
                  trash_location.append(coords)
                if (distance_sensor1.distance < distance_sensor2.distance):
                    while (distance_sensor1.distance < 0.4):
                        print("----------------------------------------")
                        print("TURNING RIGHT")
                        print("CURRENT DIRECTION = ", direction)
                        turn_right()
                        direction = dir_after_right()
                        print("NEW DIRECTION = ", direction)
                        print("----------------------------------------")
                    vehicle.stop()
                    time.sleep(1)
                    start_move = time.time()
                    prev_funct()
                    break
                    # KILL MOTION/MOMENTUM FOR CONTROL
                else:
                    while (distance_sensor2.distance < 0.4):
                        print("----------------------------------------")
                        print("TURNING LEFT")
                        print("CURRENT DIRECTION = ", direction)
                        turn_left()
                        direction = dir_after_left()
                        print("NEW DIRECTION = ", direction)
                    vehicle.stop()
                    time.sleep(1)
                    start_move = time.time()
                    prev_funct()
                    break

    
if __name__ == '__main__':

    """ distance_sensor1 = distance_sensor_module.DistanceSensor({
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
            }) """
        
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
    
    
    vehicle.stop()
    