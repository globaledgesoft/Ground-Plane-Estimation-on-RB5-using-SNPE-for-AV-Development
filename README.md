# Ground Plane Estimation on RB5 using SNPE for AV Development
## Introduction
This project is intended to build and deploy an Autonomous Vehicle Development Use case for the ground plane estimation on RB5 from RGB camera frames using Deep Learning approach. The implemented use case uses the Qualcomm’s ISP camera on RB5 for the taking the camera feed as an input using GStreamer pipeline & returns the ground plane estimation with help of Deep Learning model.


## Prerequisites 
- A Linux host system with Ubuntu 18.04. 

- Install Android Platform tools (ADB, Fastboot) 

- Download and install the SDK Manager for RB5. Link as given below https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools  

- Flash the RB5 firmware image on to the RB5 

- Setup the Network on RB5 using below link 
https://developer.qualcomm.com/qualcomm-robotics-rb5-kit/quick-start-guide/qualcomm_robotics_rb5_development_kit_bring_up/set-up-network
 
- Setup OpenCV from source for C++ support on RB5. 

- Install Python3.6 on RB5 

## Steps to Setup the Ground Plane Estimation on RB5 using GroundNet 
### Installing Dependencies

- OpenCV & Pybind11 installation on RB5 

  ```sh
   $ adb shell
   sh4.4 # python3 -m pip install --upgrade pip
   sh4.4 # python3 -m pip install opencv-python 
   sh4.4 # apt update && apt install python3-pybind11
  ```

- Setting up the SNPE Libraries on RB5

  1. Copy the SNPE header files & runtime libraries for `aarch64-ubuntu-gcc7.5` on RB5 from host system using ADB
    ```sh
        $ adb push <SNPE_ROOT>/include/ /data/snpe/include/
        $ adb push <SNPE_ROOT>/lib/aarch64-ubuntu-gcc7.5/* /data/snpe/
        $ adb push <SNPE_ROOT>/lib/dsp/* /data/snpe/
    ```
    `Note: If device is connected via SSH, please use scp tool for copying the SNPE runtime libraries in /data/snpe folder on RB5.`

  2. Open the terminal of RB5 and append the lines given below at the end of `~/.bashrc` file.
    ```sh
        sh4.4 # export PATH=$PATH:/data/snpe/
        sh4.4 # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/snpe/
        sh4.4 # export ADSP_LiBRARY_PATH="/data/snpe;/system/lib/efsa/adsp;system/vendor/lb/rfsa/adsp;/dsp"
    ```

  3. Run the command given below to reinitialize the RB5's terminal session
    ```sh
        sh4.4 # source ~/.bashrc
    ```

## Model Conversion to DLC using SNPE on Host System

1. Before running the steps given below, make sure that SNPE SDK & TensorFlow has been installed on the host system. 

2. Clone the project on host and follow below steps to convert TensorFlow model to DLC  
    ```sh
    $ git clone <source repository> 
    $ cd Ground_Plane_Estimation 
    ```
3. Run below command from <SNPE_ROOT> for Initializing the SNPE SDK for the TensorFlow Environment
    ```sh
    $ source bin/envsetup.sh -t <TENSORFLOW_DIR>
    ```
      `Note: <TENSORFLOW_DIR> is path to TensorFlow package.`

4. Navigate to SNPE 1.68.0.3932 folder in host system and run below command to convert model (Depth_model.pb) to DLC 
    ```sh
    $ ~/snpe-1.68.0.3932 $ snpe-tensorflow-to-dlc --input_network /models/ground_plane_model.pb --input_dim images “1,512,512,3” --out_node “SemanticPredictions” --output_path ground_plane.dlc 
    ```
5. Push the converted ground_plane.dlc into RB5  Ground_Plane_Estimation project    directory, by running following command 
    ```sh
    $ adb push <ROOT_PATH>/ground_plane.dlc /Ground_Plane_Estimation/models/ 
    ```

## Building the SNPE Python Wrapper for Ground Plane Estimation on RB5 using GroundNet 

1. Push the cloned source code from host system to RB5 in the Ground Plane Estimation on RB5 using GroundNet project directory. 
    ```sh
    $ adb push <host_system>/Ground_Plane_Estimation/ /home/Ground_Plane_Estimation
    ```
2. Navigate to Ground_Plane_Estimation folder location. 
    ```sh
    sh4.4 # cd Ground_Plane_Estimation
    ```
3.	Run the command below  in order to build the shared library for Python wrapper of the SNPE.
    ```sh
    sh4.4 # g++ -std=c++11 -fPIC -shared -o qcsnpe.so src/qcsnpe.cpp -I include/ -I /data/snpe/include/zdl/ -I /usr/include/python3.6m/ -I /usr/local/lib/python3.6/dist-packages/pybind11/include -L /data/snpe/ -lSNPE `pkg-config --cflags --libs opencv`
    ```
4. Above command will generate qcsnpe.so file in the Ground Plane Estimation on RB5 using GroundNet folder on RB5. 

      `Note: Give the proper path to qcsnpe.cpp and to other include files` 

## Running Ground Plane Estimation on RB5 using GroundNet

1.	Open shell of RB5 and create the gstreamer pipeline for the camera using qtiqmmfsrc element for input source.  
    ```sh
    sh4.4 # gst-launch-1.0 -e qtiqmmfsrc name=qmmf ! video/x-h264,     format=NV12, width=1280, height=720, framerate=30/1 ! h264parse config-interval=1 ! mpegtsmux name=muxer ! queue ! tcpserversink port=8080 host=localhost
    ```
      `Note: Please keep running the above pipeline in seperate terminal.`

2.	Parallelly in another terminal go to the Ground_Plane_Estimation project directory on RB5 shell and run the Depth Perception application.
    ```sh
    sh4.4 # cd  Ground_Plane_Estimation
    sh4.4 # python3 ground_plane_inference.py 
    ```

