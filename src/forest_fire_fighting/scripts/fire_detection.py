#!/usr/bin/env python3

import cv2

# check python version
import sys
print("Python version: " + sys.version)

import torch
import glob
import os
from numpy import asarray
import PIL.Image as Image
# import pandas
import csv
from ultralytics import YOLO

import time
import math
import pathlib
import datetime

import rospy

# callback function to show the image using opencv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

get_path = pathlib.Path.cwd()
date = datetime.datetime.now().strftime("%Y%m%d")

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
# model = YOLO("YoloWeights/yolov8n.pt")

# to get the parameter:
# yolo task=detect mode=train model=yolov8n.pt data=AVITAGS_NAVLAB20230930-1/data.yaml epochs=30 imgsz=640
classNames = ["Wildfire Spot"]
model = YOLO("/home/qin/m300_ws/src/forest_fire_fighting/scripts/YoloWeights/best.pt")
# model = YOLO("/home/qin/m300_ws/src/forest_fire_fighting/scripts/YoloWeights/yolov8n.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# params for recording the yolo processed video
# frame_size = (640, 480)  # width, height
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# print("[INFO]", date)
# save_path = 'home/nav/dev/datasets/videos/20231016/'

# import Detection2DArray
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray

def callback(image, pub):
    try:
        frame = CvBridge().imgmsg_to_cv2(image, "bgr8")
        # convert color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("original", frame)
        cv2.waitKey(1)
        results = model(frame, stream=True)
        ros_boxes = BoundingBoxArray()
        ros_boxes.header = image.header
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                # print(f"======> The Coordinate x_min y_min x_max y_x=max\n", x1, y1, x2, y2)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # print("======> Confidence", confidence)

                # class_name
                cls = int(box.cls[0])
                # print("======> Class Name", classNames[cls])

                # object detials
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 1

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
                cv2.imshow('processed', frame)
                cv2.waitKey(1)
                # add results to the Detection2DArray
                ros_box = BoundingBox()
                # ros_box.header = image.header
                ros_box.pose.position.x = (x1 + x2) / 2
                ros_box.pose.position.y = (y1 + y2) / 2
                ros_box.dimensions.x = x2 - x1
                ros_box.dimensions.y = y2 - y1
                ros_box.dimensions.z = 0.0
                ros_boxes.boxes.append(ros_box)
            pub.publish(ros_boxes)
    except CvBridgeError as e:
        print(e)


def listener():
    rospy.init_node('listener', anonymous=True)
    # publish the detection bounding boxes using jsk_recognition_msgs/Detection2DArray
    pub = rospy.Publisher('/bounding_boxes/fire_spots', BoundingBoxArray, queue_size=10)
    # rospy.Subscriber("/dji_osdk_ros/main_wide_RGB", Image, callback, pub)
    rospy.Subscriber("/dji_osdk_ros/main_camera_images", Image, callback, pub)
    rospy.spin()


if __name__ == '__main__':
    listener()
    cv2.destroyAllWindows()