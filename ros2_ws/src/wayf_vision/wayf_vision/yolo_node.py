import os
import numpy as np
import cv2
from collections import defaultdict
import time
import argparse

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped, PoseArray
from tf2_geometry_msgs import do_transform_point

from wayf_vision.bbox_utils import create_bbox3d_array, depth2PointCloud, detections_to_rviz_marker, detections_to_pose_array, add_box

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
from ament_index_python.packages import get_package_share_directory

class WayfindingYOLONode(Node):
    def __init__(self):
        super().__init__('wayfinding_yolo_node')

        self.declare_parameter("rotate", True)
        self.declare_parameter("namespace", '/camera')
        self.declare_parameter("readCameraInfo", True)
        self.declare_parameter("base_id", 'laser')

        namespace = self.get_parameter("namespace").get_parameter_value().string_value
        self.rotate = self.get_parameter("rotate").get_parameter_value().bool_value
        self.params = [302.86871337890625, 302.78631591796875, 212.50067138671875, 125.79319763183594]
        self.description_sub = self.create_subscription(
            CameraInfo,
            namespace + '/color/camera_info',
            self.camera_info_callback,
            10)
        self.readCameraInfo = self.get_parameter("readCameraInfo").get_parameter_value().bool_value

        self.frame_id = 'camera_link'
        self.base_id = self.get_parameter("base_id").get_parameter_value().string_value
        self.depth_scale = 0.001  # Assuming depth is in mm, convert to meters

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest_transform = None
        self.timer = self.create_timer(0.02, self.lookup_latest_transform)

        self.image_dict = defaultdict(lambda : [None, None])  # Dictionary to hold (color_image, depth_image) pairs
        self.depth_subscription = self.create_subscription(
            Image,
            namespace + '/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.depth_subscription # prevent unused variable warning
        self.image_subscription = self.create_subscription(
            Image,
            namespace + '/color/image_raw',
            self.image_callback,
            10)
        self.image_subscription # prevent unused variable warning
        self.bboximage_pub = self.create_publisher(Image, namespace + '/yolo/image', 10)
        self.rviz_pub = self.create_publisher(Marker, namespace + '/yolo/rviz', 10)
        self.rviz_bbox3d_pub = self.create_publisher(MarkerArray, namespace + '/yolo/rviz_bbox3d', 10)
        self.posearray_pub = self.create_publisher(PoseArray, namespace + '/yolo/detections', 10)
        self.prev_bbox3dlen = 0

        self.bridge = CvBridge()
        package_path = get_package_share_directory('wayf_vision')
        self.model = YOLO(os.path.join(package_path, 'models/yolov8m-seg.pt'))

        self.get_logger().info(f"Initialized WayfindingYOLONode with namespace: {namespace}")

    def camera_info_callback(self, camera_info_msg):
        if not self.readCameraInfo:
            return
        self.params = [camera_info_msg.k[0], camera_info_msg.k[4], camera_info_msg.k[2], camera_info_msg.k[5]]
        self.frame_id = camera_info_msg.header.frame_id
        self.get_logger().info(f"Camera intrinsics set: fx={self.params[0]}, fy={self.params[1]}, " \
                                + f"cx={self.params[2]}, cy={self.params[3]}, frame_id={self.frame_id}")
        self.readCameraInfo = False  # Set to False to avoid re-reading camera info

    def depth_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        timestamp_key = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        self.image_dict[timestamp_key][1] = cvImg
        if self.image_dict[timestamp_key][0] is not None:
            self.yolo_callback(image_msg.header.stamp)
            self.image_dict[timestamp_key] = [None, None]

    def image_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        timestamp_key = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        self.image_dict[timestamp_key][0] = cvImg
        if self.image_dict[timestamp_key][1] is not None:
            self.yolo_callback(image_msg.header.stamp)
            self.image_dict[timestamp_key] = [None, None]

    def yolo_callback(self, timestamp : Time):
        func_start = time.time()
        timestamp_key = timestamp.sec + timestamp.nanosec * 1e-9
        color_img, depth_img = self.image_dict[timestamp_key]
        if color_img is None or depth_img is None:
            self.get_logger().warn(f"Missing images for timestamp")
            return

        output_img, humans = self.run_detection(color_img, self.rotate)
        
        # Publish image with bounding boxes
        image_msg = self.bridge.cv2_to_imgmsg(output_img, encoding='rgb8')
        image_msg.header.stamp = timestamp
        image_msg.header.frame_id = self.frame_id
        self.bboximage_pub.publish(image_msg)
        # self.get_logger().info(f"Publishing image took : {time.time() - func_start:.4f} seconds")
        
        # Publish rviz markers
        centers, bboxes3d = self.process_human_points(color_img, depth_img, humans)

        self.posearray_pub.publish(detections_to_pose_array(centers, timestamp, self.base_id))

        dets_msg = detections_to_rviz_marker(centers, timestamp, self.base_id)
        self.rviz_pub.publish(dets_msg)

        bboxes3d_msg = create_bbox3d_array(bboxes3d, self.frame_id, timestamp, num_prev=self.prev_bbox3dlen)
        self.prev_bbox3dlen = len(bboxes3d)
        self.rviz_bbox3d_pub.publish(bboxes3d_msg)
        if len(humans) > 0:
            self.get_logger().info(f"Published {len(bboxes3d)} 3D bounding boxes")
            self.get_logger().info(f"Total Processing Time: {time.time() - func_start:.4f} seconds")

    def run_detection(self, color_img, rotate=False):
        """
        Run YOLO detection on the provided color and depth images.
        Returns the color image with bounding boxes drawn and a list of detected humans.
        If rotate is True, the color image will be rotated 90 degrees clockwise.
        """
        if rotate:
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
            image_height, image_width = color_img.shape[:2]

        yolo_start = time.time()
        results = self.model(color_img, verbose=False)
        self.get_logger().info(f"YOLO inference took {time.time() - yolo_start:.4f} seconds")

        if rotate:
            # Rotate the color image back to its original orientation
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        humans = []
        if results == None or len(results) == 0 or len(results[0].masks) == 0:
            self.get_logger().info("No detections found")
            return color_img, humans
            
        # Iterate through results to draw bounding boxes and labels
        for mask, box in zip(results[0].masks.xy, results[0].boxes):
            if CLASSES[int(box.cls[0].item())].lower() not in ['person']:
                continue
            bbox = box.xyxy.cpu().numpy().flatten().tolist()
            if rotate:
                # Rotate the points of the mask and bounding box counterclockwise
                mask = np.array(mask)
                mask[:, 0], mask[:, 1] = mask[:, 1], image_width-1-mask[:, 0]  # Rotate points
                bbox[0], bbox[1] = bbox[1], image_width-1-bbox[0]
                bbox[2], bbox[3] = bbox[3], image_width-1-bbox[2]
            humans.append((mask, bbox))

        if len(humans) > 0:
            # Run NMS to filter out overlapping detections
            boxes = np.array([box for (_, box) in humans])
            scores = np.array([0.9] * len(boxes))  # Assuming a constant score for simplicity
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.7)
            humans = [humans[i] for i in indices.flatten()]


            self.get_logger().info(f"{len(humans)} found: {[box for (mask, box) in humans]}")

            for (mask, box) in humans:
                cv2.fillPoly(color_img, np.int32([mask]), (0, 255, 0))
                color_img = add_box(color_img, box)

        return color_img, humans

    def process_human_points(self, color_img, depth_img, humans):
        """
        Create a 3D bounding box and center point from the point cloud.
        Returns a list of bounding box coordinates.
        """
        centers = []
        bboxes3d = []
        for (mask, box) in humans:
            pcd_mask = np.zeros_like(depth_img, dtype=np.uint8)
            cv2.fillPoly(pcd_mask, np.int32([mask]), 1)
            pointcloud = depth2PointCloud(depth_img, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR), self.depth_scale,
                                10.0, pcd_mask,
                                self.params)
            if pointcloud.is_empty():
                self.get_logger().warn("Point cloud is empty, skipping this detection")
                continue
            # filter out outliers using z-score
            points = np.asarray(pointcloud.points)
            z_scores = np.abs((points[:, 2] - np.mean(points[:, 2])) / np.std(points[:, 2]))
            valid_indices = np.where(z_scores < 3)[0]
            pointcloud = pointcloud.select_by_index(valid_indices)

            if pointcloud.is_empty():
                self.get_logger().warn("Point cloud is empty after filtering, skipping this detection")
                continue
            
            # Compute the bounding box of the point cloud
            x,y,z = pointcloud.get_center()
            
            point = PointStamped()
            point.point.x = x
            point.point.y = y
            point.point.z = z
            self.get_logger().info(f"Point before transform: {point.point.x}, {point.point.y}, {point.point.z}")
            if self.latest_transform is not None:
                point = do_transform_point(point, self.latest_transform)
                self.get_logger().info(f"Transformed point: {point.point.x}, {point.point.y}, {point.point.z}")
            centers.append([point.point.x, point.point.y])

            x_min, y_min, z_min = pointcloud.get_min_bound()
            x_max, y_max, z_max = pointcloud.get_max_bound()
            bboxes3d.append([x_min, y_min, z_min, x_max, y_max, z_max])
            self.get_logger().info(f"bbox3d: {x_min, y_min, z_min, x_max, y_max, z_max}, center: {x, y, z}")

        return centers, bboxes3d
    
    def lookup_latest_transform(self):
        try:
            # Use time=0 to get the latest available transform
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                source_frame=self.frame_id,
                target_frame=self.base_id,
                time=Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            self.latest_transform = transform
            # self.get_logger().info(f"Cached latest transform:\n{transform}")
        except Exception as e:
            # self.get_logger().warn(f"Transform unavailable: {e}")
            return

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
