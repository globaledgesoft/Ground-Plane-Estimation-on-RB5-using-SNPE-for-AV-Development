#!/usr/bin/env python3.6
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
import os
import time
import argparse
import glob
import qcsnpe as qc
import cv2

CPU = 0
GPU = 1
DSP = 2


def static_ROI(frame):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    roi_corners = np.array([[(180,720), (432,300), (1000,300), (1200,720)]], dtype=np.int32)
    channel_count = frame.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    
    return mask


def dynamic_ROI(frame, road_mask):
    # Finding Contours
    contours, _ = cv2.findContours(road_mask, 1, 2)
    contours = sorted(contours, key=cv2.contourArea)

    # Convex Hull
    hull = [cv2.convexHull(contours[i], False) for i in range(len(contours))]

    # Fillpoly
    drawing = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), np.uint8)
    cv2.fillPoly(drawing, pts =hull, color=(255,255,255))
    cv2.fillPoly(drawing, pts =contours, color=(255,255,255))

    # Preprocessing: Erosion, Dialation, and Resizing 
    img_dilation = cv2.dilate(drawing, np.ones((5,5), np.uint8), iterations=1)
    img_erosion = cv2.erode(img_dilation, np.ones((5,5), np.uint8), iterations=1)
    resized_mask = cv2.resize(img_erosion, (frame.shape[1], frame.shape[0]))

    return resized_mask
    
    
def ROI(frame, seg):
    # Define Road Mask
    # print(seg)
    road_mask = 1 - np.where(seg==0, seg, 1).astype('uint8')
    
    try:
        # Dynamic ROI
        print("dynamic")
        ROI_output = dynamic_ROI(frame, road_mask)
        return ROI_output * frame
    except:
        # Static ROI
        print("static")
        ROI_output = static_ROI(frame)
        return ROI_output * frame


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [0,0,0]
  colormap[3] = [0,0,0]
  colormap[4] = [0,0,0]
  colormap[5] = [0,0,0]
  colormap[6] = [0,0,0]
  colormap[7] = [0,0,0]
  colormap[8] = [0,0,0]
  colormap[9] = [152, 251, 152]
  colormap[10] = [0,0,0]
  colormap[11] = [0,0,0]
  colormap[12] = [0,0,0]
  colormap[13] = [0,0,0]
  colormap[14] = [0,0,0]
  colormap[15] = [0,0,0]
  colormap[16] = [0,0,0]
  colormap[17] = [0,0,0]
  colormap[18] = [0,0,0]
  return colormap

def label_to_color_image(label):
  
  colormap = create_cityscapes_label_colormap()
  # print(label)
  return colormap[label]


def vis_segmentation(image, seg_map):
  
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  return seg_image
  


def main():
    print()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_folder", default=None, help="image_folder")
    ap.add_argument("-v", "--vid", default=None,help="cam/video_path")
    args = vars(ap.parse_args())

    im_folder_path =  args["img_folder"]
    vid = args["vid"]

    if vid == None and im_folder_path == None:
        print("required command line args atleast ----img_folder <image folder path> or --vid <cam/video_path>")
        exit(0)

    LABEL_NAMES = np.asarray([
      'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
      'bus', 'train', 'motorcycle', 'bycycle'])


    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    model_path = "model/ground_plane.dlc"

    out_layers = ["SemanticPredictions:0"]
    model = qc.qcsnpe(model_path, out_layers, CPU)

    # imgae inference
    if im_folder_path is not None:
        for image_path in glob.glob(im_folder_path + '/*.jpg'):

            original_im = cv2.imread(image_path)

            # inferences DeepLab model
            start_time = time.time()
            original_im = cv2.resize(original_im,(512,512))
            output_img = model.predict(original_im)
            seg_map = output_img["SemanticPredictions:0"]
            resized_im = cv2.resize(image[...,::-1], (512, 512))
            
            
            ellapsed_time = time.time() - start_time
            print("Ellapsed time: " + str(ellapsed_time) + "s")

            masked_image = ROI(np.array(resized_im), seg_map)
            cv2.imshow("masked_image", masked_image)
            cv2.waitKey(0)
            
            # show inference result
            map_im = vis_segmentation(resized_im, seg_map)
            cv2.imshow("seg_image", map_im)
            cv2.waitKey(0)

    # video inference
    if vid is not None:
        if vid == "cam":
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(vid)

        out = cv2.VideoWriter("output.mp4",  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            10, (512, 512)
                            )

        while True:
            ret, image = video_capture.read()  # frame shape 640*480*3
            if ret:
                
                original_im = image.copy()
                # inferences DeepLab model
                start_time = time.time()
                original_im = cv2.resize(original_im,(512,512))
                output_img = model.predict(original_im)
                seg_map = output_img["SemanticPredictions:0"]
                resized_im = cv2.resize(image[...,::-1], (512, 512))
                
                ellapsed_time = time.time() - start_time
                print("Ellapsed time: " + str(ellapsed_time) + "s")

                masked_image = ROI(np.array(resized_im), seg_map)
                #cv2.imshow("masked_image", masked_image)
                
                # show inference result
                #map_im = vis_segmentation(resized_im, seg_map)
                #cv2.imshow("seg_image", map_im)
                masked_image = cv2.resize(masked_image,(512,512))

                out.write(masked_image.astype(np.uint8))

                k = cv2.waitKey(1)
                if k == ord('q'):
                    break
            else:
              break

        video_capture.release()
        out.release()

if __name__ == '__main__':
    main()
