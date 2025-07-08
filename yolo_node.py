import numpy as np
import cv2
from collections import defaultdict
import time

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
import open3d as o3d

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
from geometry_msgs.msg import Point

def add_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def detections_to_rviz_marker(dets_xy):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "yolo_ros"
    msg.id = 0
    msg.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    # blue color
    msg.color.b = 1.0
    msg.color.a = 1.0

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy in dets_xy:
        for i in range(len(xy_offsets) - 1):
            # start point of a segment
            p0 = Point()
            p0.x = float(d_xy[0] + xy_offsets[i, 0])
            p0.y = float(d_xy[1] + xy_offsets[i, 1])
            p0.z = 0.0
            msg.points.append(p0)

            # end point
            p1 = Point()
            p1.x = float(d_xy[0] + xy_offsets[i + 1, 0])
            p1.y = float(d_xy[1] + xy_offsets[i + 1, 1])
            p1.z = 0.0
            msg.points.append(p1)

    return msg

def depth2PointCloud(depth, rgb, depth_scale, clip_distance_max, mask, intrinsics):
    [fx, fy, cx, cy] = intrinsics
    depth = depth * depth_scale # 1000 mm => 0.001 meters
    rows,cols  = depth.shape

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)
    z = depth 
    x =  z * (c - cx) / fx
    y =  z * (r - cy) / fy

    depth = depth[np.where(mask>0)]

    valid = (depth > 0) & (depth < clip_distance_max)
    valid = np.ravel(valid)
    
    z = np.ravel(z[np.where(mask>0)])[valid]
    x = np.ravel(x[np.where(mask>0)])[valid]
    y = np.ravel(y[np.where(mask>0)])[valid]
    
    r = np.ravel(rgb[:,:,2][np.where(mask>0)])[valid]
    g = np.ravel(rgb[:,:,1][np.where(mask>0)])[valid]
    b = np.ravel(rgb[:,:,0][np.where(mask>0)])[valid]
    
    pointsxyzrgb = np.dstack((x, y, z, r, g, b))
    pointsxyzrgb = pointsxyzrgb.reshape(-1,6)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pointsxyzrgb[:,:3])
    if(pointsxyzrgb.shape[1]>3):
        rgb_t = pointsxyzrgb[:,3:]
        pc.colors = o3d.utility.Vector3dVector(rgb_t.astype(float) / 255.0)
    pc = pc.voxel_down_sample(voxel_size=0.00001)

    return pc

class WayfindingYOLONode(Node):
    def __init__(self, namespace='wayfinding/camera'):
        self.fx = 302.86871337890625 # Change this to use the ros topic later
        self.fy = 302.78631591796875
        self.cx = 212.50067138671875
        self.cy = 125.79319763183594
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
        self.rviz_pub = self.create_publisher(Marker, namespace + '/yolo/rviz', 10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8m-seg.pt', verbose=False)
    
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

        yolo_start = time.time()
        results = self.model(color_img)
        # self.get_logger().info(f"YOLO inference took {time.time() - yolo_start:.2f} seconds")

        humans = []
        # Iterate through NMS results to draw bounding boxes and labels
        for mask, box in zip(results[0].masks.xy, results[0].boxes):
            if CLASSES[int(box.cls[0].item())].lower() not in ['person']:
                continue
            humans.append((mask, box.xyxy.cpu().numpy().flatten().tolist()))

        if len(humans) > 0:
            self.get_logger().info(f"{len(humans)} found: {[box for (mask, box) in humans]}")

            for (mask, box) in humans:
                cv2.fillPoly(color_img, np.int32([mask]), (0, 255, 0))
                color_img = add_box(color_img, box)
        
        # Publish image with bounding boxes
        image_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='rgb8')
        image_msg.header.stamp = timestamp
        image_msg.header.frame_id = 'camera_link'
        self.bboximage_pub.publish(image_msg)
        # self.get_logger().info(f"Publishing image took : {time.time() - func_start:.4f} seconds")
        
        # Publish rviz markers
        centers = []
        for (mask, box) in humans:
            pcd_mask = np.zeros_like(depth_img, dtype=np.uint8)
            cv2.fillPoly(pcd_mask, np.int32([mask]), 1)
            pointcloud = depth2PointCloud(depth_img, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR), 1,
                                10.0, pcd_mask,
                                [self.fx, self.fy, self.cx, self.cy])
            if pointcloud.is_empty():
                self.get_logger().warn("Point cloud is empty, skipping this detection")
                continue
            x,y,z = pointcloud.get_center()
            centers.append([x,z])
        self.get_logger().info(f"Point cloud centers: {centers}")
        dets_msg = detections_to_rviz_marker(centers)
        dets_msg.header.stamp = timestamp
        dets_msg.header.frame_id = 'laser'
        self.rviz_pub.publish(dets_msg)

        self.get_logger().info(f"Total Processing Time: {time.time() - func_start:.4f} seconds")

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
