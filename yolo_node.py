import numpy as np
import cv2
from collections import defaultdict
import json
import os
# from contextlib import redirect_stdout
import torch
import time

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray

from ultralytics import YOLO
from yolov8_utils import draw_bounding_box, CLASSES
from bbox_utils import create_bbox3d

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

MIN_DEPTH = 0.3
MAX_DEPTH = 2.5

def add_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def add_mask(image, mask, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255, 144, 30, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 

    overlay_rgb = mask_image[:, :, :3]
    alpha = mask_image[:, :, 3]

    # Ensure alpha has shape (H, W, 1) for broadcasting
    alpha = alpha[:, :, np.newaxis]
    # print(image)
    # Blend the images
    blended = (alpha * overlay_rgb + (1 - alpha) * image).astype(np.uint8)
    return blended

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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
    
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
        func_start = time.time()
        timestamp_key = timestamp.sec + timestamp.nanosec * 1e-9
        color_img, depth_img = self.image_dict[timestamp_key]
        if color_img is None or depth_img is None:
            self.get_logger().warn(f"Missing images for timestamp")
            return
        # self.get_logger().info(f"Running YOLO for timestamp {timestamp}")
        bbox_msg = MarkerArray()

        yolo_start = time.time()
        results = self.model(color_img)
        self.get_logger().info(f"YOLO inference took {time.time() - yolo_start:.2f} seconds")

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
        if len(boxes) == 0:
            # self.get_logger().warn(f"No bounding boxes detected for timestamp {timestamp}")
            return
        # Apply NMS (Non-maximum suppression)
        nms_start = time.time()
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        self.get_logger().info(f"Detected {len(result_boxes)} bounding boxes in {time.time() - nms_start:.2f} seconds")

        humans = []
        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            if CLASSES[class_ids[result_boxes[i]]].lower() not in ['person']:
                continue
            index = result_boxes[i]
            box = boxes[index]
            humans.append(box)

        if len(humans) > 0:
            self.get_logger().info(f"{len(humans)} found: {humans}")
            
            mask_start = time.time()
            self.predictor.set_image(color_img)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=humans,
                multimask_output=False,
            )
            self.get_logger().info(f"Mask prediction took {time.time() - mask_start:.2f} seconds")

            for box in humans:
                color_img = add_box(color_img, box)

            # for box, mask in zip(humans, masks):
            #     # find min and max depth in the bounding box area
            #     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            #     if(not np.any(mask)):
            #         self.get_logger().warn(f"Bounding box {i} has no valid depth values")
            #         continue

            #     color_img = add_mask(color_img, mask, borders=True) 
        
        # Publish image with bounding boxes
        image_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='rgb8')
        image_msg.header.stamp = timestamp
        image_msg.header.frame_id = 'camera_link'
        self.bboximage_pub.publish(image_msg)

        # Publish 3D bounding boxes
        # self.bbox3d_pub.publish(bbox_msg)
        # self.get_logger().info(f"Published {len(bbox_msg.markers)} 3D bounding boxes for timestamp {timestamp}")
        self.get_logger().info(f"Processing time: {time.time() - func_start:.2f} seconds")

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
