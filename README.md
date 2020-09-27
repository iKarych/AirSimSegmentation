# AirSimSegmentation

To run this project, you need to install AirSim package and Unreal Engine (or a ready binary).

#####1) Run collect_images.py, to collect dataset.
A collected dataset can be seen in neighbor_set from binary [Neighborhood](https://github.com/microsoft/AirSim/releases/tag/v1.3.1-linux) . This script controls the drone, flies it over the scene, takes pictures using front and back camera and saves RGB images to "image" folder,  and segmentation images to "label"

![](/home/ikarych/AirSim/Unreal/Environments/Neighborhood/screen.png "AirSim in Unreal Engine") 

#####2) Train UNet - train_unet.py
First, move some RGB and segmentation images to test folder. The script sets up the UNet network, preporcesses inputs and outputs, and trains the network. Weights are saved in unet_neighbor.hdf5

Then a validation is performed on the test dataset. "validated" folder contains: original image, segmentation image, predicted segmentation image

![ ](/home/ikarych/Desktop/unet/AirSimSegmentation/validated/images/image_00004.png  "RGB image") ![ ](/home/ikarych/Desktop/unet/validated/label/label_00004.png  "Segmentation Image") ![ ](/home/ikarych/Desktop/unet/validated/predicted/prediction_00004.png  "Predicted segmentation" )

#####3) Control drone with UNet - control_with_unet.py
Using the trained network, an image is passed to the network, and the outcome is the percentage of the occupancy of the road on the picture. Depending on the percentage, drone's altitude is increased/decreased.