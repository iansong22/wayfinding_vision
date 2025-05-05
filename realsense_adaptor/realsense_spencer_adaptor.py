#!/usr/bin/env python

#realsense_spencer_adaptor.py
# node for connecting realsence d435i output to spencer tracking package

import rospy
import cv_bridge
import cv2
from sensor_msgs.msg import Image

CROP_W = 640
CROP_H = 480

def convertDepthEncoding(imageMsg, bridge):
    """
    convert given image from 16UC1 in mm to 32FC1 m
    using given cv bridge
    Refer https://github.com/spencer-project/spencer_people_tracking/issues/4
    """
    #convert from 16UC1 in mm to 32FC1 m
    cvImg = bridge.imgmsg_to_cv2(imageMsg)
    rotated_cvImg = cv2.rotate(cvImg, cv2.ROTATE_90_CLOCKWISE)
    
    h,w = rotated_cvImg.shape[:2]
    w_start = (w-CROP_W) // 2
    h_start = (h-CROP_H) // 2
    cropped_cvImg = rotated_cvImg[h_start:h_start+CROP_H, w_start:w_start+CROP_W]

    cvImg32F = cropped_cvImg.astype('float32') / 1000.0
    convertedImageMsg = bridge.cv2_to_imgmsg(cvImg32F)
    convertedImageMsg.header = imageMsg.header
    return convertedImageMsg

def convertMonoEncoding(imageMsg, bridge):
    """
    Resolves the error: "cv_bridge exception: [8UC1] is not a color format. but [bgr8] is."
    which arises when processing mono images from realsense using opencv.
    Note this is not required for using spencer tracking
    """
    cvImg = bridge.imgmsg_to_cv2(imageMsg)
    rotated_cvImg = cv2.rotate(cvImg, cv2.ROTATE_90_CLOCKWISE)

    h,w = rotated_cvImg.shape[:2]
    w_start = (w-CROP_W) // 2
    h_start = (h-CROP_H) // 2
    cropped_cvImg = rotated_cvImg[h_start:h_start+CROP_H, w_start:w_start+CROP_W]
    
    if len(cropped_cvImg.shape) == 2:
        encoding = 'mono8'
    elif cropped_cvImg.shape[2] == 3:
        encoding = 'bgr8' if imageMsg.encoding.lower() in ['bgr8', 'bgr'] else 'rgb8'
    convertedImageMsg = bridge.cv2_to_imgmsg(cropped_cvImg, encoding=encoding)
    convertedImageMsg.header = imageMsg.header
    # convertedImageMsg.encoding = "mono8"
    return convertedImageMsg


def runAdaptor():
    rospy.init_node('realsense_spencer_adaptor')
    bridge = cv_bridge.CvBridge()
    
    pubDepth = rospy.Publisher('depth/image_rect', Image, queue_size=10)
    callbackDepth = lambda imgMsg : pubDepth.publish(convertDepthEncoding(imgMsg, bridge))
    subDepth = rospy.Subscriber('depth/image_rect_raw', Image, callbackDepth)
    # subDepth = rospy.Subscriber('aligned_depth_to_color/image_raw', Image, callbackDepth)

    pubInfra = rospy.Publisher('rgb/image_rect_color', Image, queue_size=10)
    callbackInfra = lambda imgMsg : pubInfra.publish(convertMonoEncoding(imgMsg, bridge))
    subInfra = rospy.Subscriber('color/image_raw', Image, callbackInfra)

    rospy.spin()

if __name__ == '__main__':
    runAdaptor()
