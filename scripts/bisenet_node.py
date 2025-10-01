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
from ros2_ws.src.wayf_vision.wayf_vision.bbox_utils import create_bbox3d_array, depth2PointCloud, detections_to_rviz_marker, add_box

import torch
from BiSeNet.lib.models import model_factory
import BiSeNet.lib.data.transform_cv2 as T
from BiSeNet.configs import set_cfg_from_file

class WayfindingBisenetNode(Node):
    def __init__(self, namespace, rotate=False, readCameraInfo=True):
        super().__init__('wayfinding_bisenet_node')
        self.rotate = rotate
        self.params = [302.86871337890625, 302.78631591796875, 212.50067138671875, 125.79319763183594]
        self.description_sub = self.create_subscription(
            CameraInfo,
            namespace + '/color/camera_info',
            self.camera_info_callback,
            10)
        self.readCameraInfo = readCameraInfo

        self.frame_id = 'camera_link'
        self.depth_scale = 0.001  # Assuming depth is in mm, convert to meters

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
        self.bboximage_pub = self.create_publisher(Image, namespace + '/bisenet/image', 10)
        self.rviz_pub = self.create_publisher(Marker, namespace + '/bisenet/rviz', 10)
        self.rviz_bbox3d_pub = self.create_publisher(MarkerArray, namespace + '/bisenet/rviz_bbox3d', 10)
        self.bbox3d_pub = self.create_publisher(Float32MultiArray, namespace + '/bisenet/bbox3d', 10)

        self.bridge = CvBridge()
        cfg = set_cfg_from_file(args.config)

        # define model
        net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
        net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
        net.eval()

        self.get_logger().info(f"Initialized WayfindingBisenetNode with namespace: {namespace}")

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
            self.run_yolo(image_msg.header.stamp)
            self.image_dict[timestamp_key] = [None, None]

    def image_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        timestamp_key = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
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

        output_img, humans = self.run_detection(color_img, self.rotate)
        
        # Publish image with bounding boxes
        image_msg = self.bridge.cv2_to_imgmsg(output_img, encoding='rgb8')
        image_msg.header.stamp = timestamp
        image_msg.header.frame_id = self.frame_id
        self.bboximage_pub.publish(image_msg)
        # self.get_logger().info(f"Publishing image took : {time.time() - func_start:.4f} seconds")
        
        # Publish rviz markers
        centers, bboxes3d = self.process_human_points(color_img, depth_img, humans)

        self.bbox3d_pub.publish(Float32MultiArray(data=np.array(bboxes3d).flatten().tolist()))

        dets_msg = detections_to_rviz_marker(centers)
        dets_msg.header.stamp = timestamp
        dets_msg.header.frame_id = self.frame_id
        self.rviz_pub.publish(dets_msg)
        self.rviz_bbox3d_pub.publish(create_bbox3d_array(bboxes3d, self.frame_id, timestamp))
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
        
        results = self.model(color_img, verbose=False)

        # prepare data
        to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

        humans = []
        # Iterate through NMS results to draw bounding boxes and labels
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

        if rotate:
            # Rotate the color image back to its original orientation
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if len(humans) > 0:
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
            centers.append([x,z])
            x_min, y_min, z_min = pointcloud.get_min_bound()
            x_max, y_max, z_max = pointcloud.get_max_bound()
            bboxes3d.append([x_min, y_min, z_min, x_max, y_max, z_max])
            self.get_logger().info(f"bbox3d: {x_min, y_min, z_min, x_max, y_max, z_max}, center: {x, y, z}")

        return centers, bboxes3d

def main(namespace, rotate, args=None):
    rclpy.init(args=args)

    node = WayfindingBisenetNode(namespace=namespace, rotate=rotate)

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='/camera',
                        help='Namespace for the camera topic')
    parser.add_argument('--rotate', action='store_true', default=False,
                        help='Rotate the input image 90 degrees clockwise before processing')
    args = parser.parse_args()
    namespace = args.namespace
    rotate = args.rotate
    main(namespace=namespace, rotate=rotate)
