import numpy as np
import cv2
from collections import defaultdict
import json
import os
from contextlib import redirect_stdout

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray

from ultralytics import YOLO
from yolov8_utils import draw_bounding_box, CLASSES
from bbox_utils import create_bbox3d

MIN_DEPTH = 0.3
MAX_DEPTH = 2.5

class WayfindingYOLONode(Node):
    def __init__(self, namespace='wayfinding/camera'):
        with open("camera_parameters.json", "r") as f:
            data = json.load(f)
        self.fx = data["fy"]
        self.fy = data["fx"]
        self.cx = data["cy"]
        self.cy = data["cx"]
        super().__init__('wayfinding_rotation_node')
        self.image_dict = defaultdict(lambda : [None, None])  # Dictionary to hold (color_image, depth_image) pairs
        self.depth_subscription = self.create_subscription(
            Image,
            namespace + '/depth/image',
            self.depth_callback,
            10)
        self.depth_subscription # prevent unused variable warning
        self.image_subscription = self.create_subscription(
            Image,
            namespace + '/color/image',
            self.image_callback,
            10)
        self.image_subscription # prevent unused variable warning
        self.bboximage_pub = self.create_publisher(Image, namespace + '/yolo/image', 10)
        self.bbox3d_pub = self.create_publisher(MarkerArray, namespace + '/yolo/box3d', 10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt', verbose=False)
    
    def depth_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        timestamp_key = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        # self.get_logger().info(f"Received depth_image at timestamp {timestamp}")
        self.image_dict[timestamp_key][1] = cvImg
        if self.image_dict[timestamp_key][0] is not None:
            self.run_yolo(image_msg.header.stamp)
            self.image_dict[timestamp_key] = [None, None]

    def image_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        timestamp_key = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        # self.get_logger().info(f"Received color_image at timestamp {timestamp}")
        self.image_dict[timestamp_key][0] = cvImg
        if self.image_dict[timestamp_key][1] is not None:
            self.run_yolo(image_msg.header.stamp)
            self.image_dict[timestamp_key] = [None, None]

    def run_yolo(self, timestamp : Time):
        timestamp_key = timestamp.sec + timestamp.nanosec * 1e-9
        color_img, depth_img = self.image_dict[timestamp_key]
        if color_img is None or depth_img is None:
            self.get_logger().warn(f"Missing images for timestamp {timestamp}")
            return
        # self.get_logger().info(f"Running YOLO for timestamp {timestamp}")
        bbox_msg = MarkerArray()

        results = self.model(color_img)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
        if len(boxes) == 0:
            # self.get_logger().warn(f"No bounding boxes detected for timestamp {timestamp}")
            return
        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        # self.get_logger().info(f"Detected {len(result_boxes)} bounding boxes for timestamp {timestamp}")


        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            if CLASSES[class_ids[result_boxes[i]]].lower() not in ['person']:
                continue
            index = result_boxes[i]
            box = boxes[index]

            # find min and max depth in the bounding box area
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            depth_area = depth_img[y1:y2, x1:x2]
            # Apply gaussian filter to depth area
            depth_area = cv2.GaussianBlur(depth_area, (5, 5), 0)
            valid_depth_mask = (depth_area > MIN_DEPTH) & (depth_area < MAX_DEPTH)
            if(not np.any(valid_depth_mask)):
                self.get_logger().warn(f"Bounding box {i} has no valid depth values")
                continue
            min_depth = float(np.min(depth_area[valid_depth_mask])) 
            average_depth = float(np.mean(depth_area[valid_depth_mask]))
            max_depth = float(np.max(depth_area[valid_depth_mask])) 
            stdev_depth = float(np.std(depth_area[valid_depth_mask]))
            self.get_logger().info(f"Bounding box {i} | min depth: {min_depth}, average depth: {average_depth}, max depth: {max_depth}, stdev: {stdev_depth}")

            # convert x1, y1, x2, y2 to 3d using intrinsic parameters
            min_x = (((x1 - self.cx) * average_depth )/ self.fx)
            min_y = (((y1 - self.cy) * average_depth )/ self.fy)
            max_x = (((x2 - self.cx) * average_depth )/ self.fx)
            max_y = (((y2 - self.cy) * average_depth )/ self.fy)
            self.get_logger().info(f"Bounding box {i} | x: ({min_x}, {max_x}), y: ({min_y}, {max_y}) with depth ({min_depth}, {max_depth})")
            
            bbox_msg.markers.append(
                create_bbox3d(
                    [min_x, min_y, average_depth-stdev_depth, max_x, max_y, average_depth+stdev_depth],
                    'camera_link',
                    timestamp
                )
            )

            draw_bounding_box(
                color_img,
                class_ids[index],
                scores[index],
                round(box[0]),
                round(box[1]),
                round(box[2]),
                round(box[3]),
            )
        
        # Publish image with bounding boxes
        image_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='rgb8')
        image_msg.header.stamp = timestamp
        image_msg.header.frame_id = 'camera_link'
        self.bboximage_pub.publish(image_msg)

        # Publish 3D bounding boxes
        self.bbox3d_pub.publish(bbox_msg)
        self.get_logger().info(f"Published {len(bbox_msg.markers)} 3D bounding boxes for timestamp {timestamp}")

def main(args=None):
    rclpy.init(args=args)

    node = WayfindingYOLONode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
