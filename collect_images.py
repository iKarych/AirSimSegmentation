import setup_path 
import airsim
import os

import sys
import time
import argparse
import math
import time

## Choose folder for collected images
dire = os.path.join("neighbor_set", "train")

## connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

## Take off and move to the start position
airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -10, 1).join()
no = 0

client.moveByRollPitchYawZAsync(math.radians(0),math.radians(0),math.radians(0), -10, 5).join()
time.sleep(2)

## Go north
for x in range(128):
    client.moveToPositionAsync(x, 0, -10, 1)

    ## Collect images
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("back_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),
        airsim.ImageRequest("back_center", airsim.ImageType.Segmentation)])

    ## Save images
    for i, response in enumerate(responses):
        if i == 0:
            filename = os.path.join(dire, "image", "image_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 1:
            filename = os.path.join(dire, "image", "image_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 2:
            filename = os.path.join(dire, "label", "segmentation_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 3:
            filename = os.path.join(dire, "label", "segmentation_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    no += 1


## Make sure you are still connected to the controller and turn
time.sleep(5)
client.enableApiControl(True)
client.moveByRollPitchYawZAsync(math.radians(0),math.radians(0),math.radians(-90), -10, 5).join() 
time.sleep(5)

## Go east
for x in range(128):
    client.moveToPositionAsync(127, x, -10, 1)

    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("back_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),
        airsim.ImageRequest("back_center", airsim.ImageType.Segmentation)])

    for i, response in enumerate(responses):
        if i == 0:
            filename = os.path.join(dire, "image", "image_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 1:
            filename = os.path.join(dire, "image", "image_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 2:
            filename = os.path.join(dire, "label", "segmentation_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 3:
            filename = os.path.join(dire, "label", "segmentation_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    no += 1
    

time.sleep(5)
client.enableApiControl(True)
client.moveByRollPitchYawZAsync(math.radians(0),math.radians(0),math.radians(180), -10, 5).join() 
time.sleep(5)

## Go south
for x in range(127,-128,-1):
    client.moveToPositionAsync(x, 127, -10, 1)

    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("back_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),
        airsim.ImageRequest("back_center", airsim.ImageType.Segmentation)])

    for i, response in enumerate(responses):
        if i == 0:
            filename = os.path.join(dire, "image", "image_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 1:
            filename = os.path.join(dire, "image", "image_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 2:
            filename = os.path.join(dire, "label", "segmentation_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 3:
            filename = os.path.join(dire, "label", "segmentation_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    no += 1


        
time.sleep(5)
client.enableApiControl(True)
client.moveByRollPitchYawZAsync(math.radians(0),math.radians(0),math.radians(90), -10, 5).join() 
time.sleep(5)

## Go west
for x in range(127,-1,-1):
    client.moveToPositionAsync(-127, x, -10, 1)

    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("back_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),
        airsim.ImageRequest("back_center", airsim.ImageType.Segmentation)])

    for i, response in enumerate(responses):
        if i == 0:
            filename = os.path.join(dire, "image", "image_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 1:
            filename = os.path.join(dire, "image", "image_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 2:
            filename = os.path.join(dire, "label", "segmentation_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 3:
            filename = os.path.join(dire, "label", "segmentation_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    no += 1
       

time.sleep(5)
client.enableApiControl(True)
client.moveByRollPitchYawZAsync(math.radians(0),math.radians(0),math.radians(0), -10, 5).join() 
time.sleep(5)

## Go north
for x in range(-127,1,1):
    client.moveToPositionAsync(x, 0, -10, 1)

    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene),
        airsim.ImageRequest("back_center", airsim.ImageType.Scene),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation),
        airsim.ImageRequest("back_center", airsim.ImageType.Segmentation)])

    for i, response in enumerate(responses):
        if i == 0:
            filename = os.path.join(dire, "image", "image_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 1:
            filename = os.path.join(dire, "image", "image_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 2:
            filename = os.path.join(dire, "label", "segmentation_front" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        elif i == 3:
            filename = os.path.join(dire, "label", "segmentation_back" + str(no).zfill(5))
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    no += 1
        
        
## Once all the data is collected, land in the starting position
print("landing...")
client.landAsync().join()

print("disarming.")
client.armDisarm(False)
