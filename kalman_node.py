import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from bbox_utils import detections_to_rviz_marker_array

import numpy as np

from kalman.model import Wayfinding_3DMOT as tracker

class KalmanTrackingNode(Node):
    def __init__(self, namespace="camera"):
        super().__init__('kalman_tracking_node')
        self.get_logger().info("Initializing Kalman Tracking Node")
        self.bbox_subscription = self.create_subscription(
            PoseArray,
            namespace + '/yolo/detections',
            self.yolo_callback,
            10)
        self.lidar_subscription = self.create_subscription(
            PoseArray,
            '/dr_spaam_detections',
            self.lidar_callback,
            10)
        self.bbox_subscription # prevent unused variable warning
        self.bbox_publisher = self.create_publisher(
            MarkerArray,
            namespace + '/kalman/tracked_boxes',
            10)
        self.tracker = tracker(vis_thres=-0.4, lidar_thres=-0.3, max_age=10, output_preds=False)
        self.colors = [
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.25, 0.0],
        ]
        self.generate_new_colors = True
        self.previous_markers_len = 0
    def yolo_callback(self, poses_msg):

        # poses are in form [x, y, z, radius]
        # tracker requires [h,w,l,x,y,z,theta]
        r = 0.4
        boxes = np.array([[
            r,
            r,
            r,
            p.position.x,
            p.position.y,
            p.position.z,
            0
        ] for p in poses_msg.poses])
        self.get_logger().info(f"Received {len(boxes)} boxes {boxes} from camera")
        dets = {"vision": boxes, "lidar": []}
        self.box_callback(dets, poses_msg.header.stamp, poses_msg.header.frame_id)

    def lidar_callback(self, poses_msg):
        # poses are in form [x, y, z, radius]
        # tracker requires [h,w,l,x,y,z,theta]
        r = 0.4
        boxes = np.array([[
            r,
            r,
            r,
            p.position.x,
            p.position.y,
            p.position.z,
            0
        ] for p in poses_msg.poses])
        self.get_logger().info(f"Received {len(boxes)} boxes {boxes} from lidar")
        dets = {"vision": [], "lidar": boxes}
        self.box_callback(dets, poses_msg.header.stamp, poses_msg.header.frame_id)

    def box_callback(self, dets, timestamp, frame_id):
        X_IDX = 3
        Y_IDX = 4
        Z_IDX = 5
        ID_IDX = 7
        results = self.tracker.track(dets)
        processed_boxes = results["results"]
        affi = results["affi"]

        if self.tracker.output_preds:
            preds = results["preds"]
            self.get_logger().info(f"Preds: {preds}")
            
        self.get_logger().info(f"affi: {affi}")

        if len(processed_boxes) > 0:
            # Process the bounding boxes with the tracker
            self.get_logger().info(f"Processed {len(processed_boxes)} bounding boxes: {processed_boxes}")
            
            boxes = []
            colors = []
            for box in processed_boxes:
                # box is in the format [h, w, l, x, y, z, theta]
                boxes.append([box[X_IDX], box[Y_IDX]])
                idx = int(box[ID_IDX])
                if len(self.colors) > idx:
                    color = self.colors[idx]
                elif self.generate_new_colors:
                    color = [np.random.rand() for _ in range(3)]
                    self.colors.append(color)
                else:
                    color = self.colors[idx % len(self.colors)]
                colors.append(color)

            # Convert processed boxes to PoseArray for publishing
            rviz_marker = detections_to_rviz_marker_array(
                boxes, timestamp, frame_id, colors=colors)
        else:
            self.get_logger().info("No boxes processed")
            rviz_marker = detections_to_rviz_marker_array([], timestamp, frame_id)

        # Add delete markers for previous boxes
        markers_len = len(rviz_marker.markers)
        if self.previous_markers_len > markers_len:
            for i in range(markers_len, self.previous_markers_len):
                delete_marker = Marker()
                delete_marker.action = Marker.DELETE
                delete_marker.ns = "yolo_ros"
                delete_marker.id = i
                delete_marker.header.frame_id = frame_id
                delete_marker.header.stamp = timestamp
                rviz_marker.markers.append(delete_marker)
        self.previous_markers_len = markers_len

        self.bbox_publisher.publish(rviz_marker)
                
def main(namespace="camera",args=None):
    rclpy.init(args=args)

    node = KalmanTrackingNode(namespace=namespace)

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Kalman Tracking Node.')
    parser.add_argument('--namespace', type=str, default='camera',
                        help='Namespace for the node (default: camera)')
    args = parser.parse_args()
    main(namespace=args.namespace)
