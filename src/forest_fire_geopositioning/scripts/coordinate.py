#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: coordinate.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-10-04
#   2023-10-10: Use camera, YOLOv8
#   2023-10-31: Use frame, YOLOv5
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: Straightly output the label bboxes' coordinate 
#
# ------------------------------------------------------------------------------
import cv2
import torch
import glob
import os
import sys
from numpy import asarray
import PIL.Image as Image
import pandas
import csv

'''###########################################################################
This part is YOLOv5-based detection, and save the bounding box coordinate
'''
# load the model information
weight_path = './YoloWeights/yolov5l/20231002/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', weight_path, force_reload = True)
# load the video which to process
frame_path = '/home/qiao/dev/giao/data/videos/20230926/frames/*.png'
imgs = glob.glob(frame_path)
imgs.sort()
save_path = '/home/qiao/dev/giao/data/videos/20230926/result'

for i, f in enumerate(imgs):
    print(f"\n======> Frame Count: ", i)
    img_bgr = cv2.imread(f)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(img_rgb)
    result = model(img_rgb)
    # box = result.pandas().xyxy[0]
    for box in result.xyxy[0]:
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
#        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 2)
#        cv2.imshow('bboxed', f) 
        print(f"======> BBoxes-Coordinate: x_min y_min x_max y_xmax\n", x1, y1, x2, y2)

    # box.to_csv(save_path + '/%04d.csv'%i)
    i += 1
cv2.destroyAllWindows()
'''###########################################################################
'''

# 20231010
'''
This part is for YOLOv8-based detection, and save the original video and processed video,
the bbox coordinates are printed
'''
# import time
# import math
# import pathlib
# import datetime
# from ultralytics import YOLO

# get_path = pathlib.Path.cwd()
# video_path = "/home/qiao/dev/datasets/videos/20231023/20231023.MP4"
# parameter_path = '/home/qiao/dev/detection_part/detection/yolov8/yolo_weights/'
# parameter = '20231023/best.pt'
# video_path = 0
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
# model = YOLO("yolo-Weights/yolov8n.pt")

# to get the parameter:
# yolo task=detect mode=train model=yolov8n.pt data=AVITAGS_NAVLAB20230930-1/data.yaml epochs=30 imgsz=640
# classNames = ["Wildfire Spot"]

# model = YOLO(parameter_path + parameter)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# params for recording the yolo processed video
# frame_size = (640, 480) # width, height
# fourcc = cv2.VideoWriter_fourcc(*"MP4V")
# date= datetime.datetime.now().strftime("%Y%m%d") 
# print("[INFO]", date)
# save_path = '/home/qiao/dev/giao/works/detecting/segmentation/yolov8/videos/20231012/'

# record_video_name = save_path + date + 'original.mp4'
# result_video_name = save_path + date+ 'bboxed.mp4'
# record_video_name = date + '_original.mp4'
# result_video_name = date+ '_bboxed.mp4'

# print("[INFO] o name and r name:", record_video_name, result_video_name)
# cam = cv2.VideoCapture(video_path)
# fps = cam.get(cv2.CAP_PROP_FPS)
# cam.set(3, 640)
# cam.set(4, 480)
# i = 0


# original = cv2.VideoWriter(record_video_name, fourcc, fps, frame_size)
# processed = cv2.VideoWriter(result_video_name, fourcc, fps, frame_size)


# while True:
#     ret, frame = cam.read()
#     cv2.imshow("original", frame)
#     print("[INFO] using camera")
#     original.write(frame)
#     print("[INFO] Recording original video")

#     results = model(frame, stream = True)
#     print("[INFO] processing the video captured")
#     
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]

            ##############################################
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            ##############################################

            # confidence
#             confidence = math.ceil((box.conf[0] * 100))/100
#             print("======> Confidence", confidence)

            # class_name
#             cls = int(box.cls[0])
#             print("======> Class Name", classNames[cls])

            # object detials
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 1

#             cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
#     cv2.imshow('processed', frame)
#     processed.write(frame)
    # print["[INFO] Recording the processed video"]

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()

# original.release()
# processed.release()

# cv2.destroyAllWindows()
