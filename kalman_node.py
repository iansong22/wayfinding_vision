import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from bbox_utils import detections_to_rviz_marker

import numpy as np

from kalman.model import AB3DMOT as tracker

class KalmanTrackingNode(Node):
    def __init__(self, namespace="camera"):
        super().__init__('kalman_tracking_node')
        self.get_logger().info("Initializing Kalman Tracking Node")
        self.bbox_subscription = self.create_subscription(
            PoseArray,
            namespace + '/yolo/detections',
            self.box_callback,
            10)
        self.bbox_subscription # prevent unused variable warning
        self.bbox_publisher = self.create_publisher(
            Marker,
            namespace + '/kalman/tracked_boxes',
            10)
        self.tracker = tracker(output_preds=True)
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
    def box_callback(self, poses_msg):
        # poses are in form [x, y, z, radius]
        # tracker requires [h,w,l,x,y,z,theta]
        r = 0.4
        X_IDX = 3
        Z_IDX = 5
        ID_IDX = 7
        boxes = np.array([[
            r,
            r,
            r,
            p.position.x,
            p.position.y,
            p.position.z,
            0
        ] for p in poses_msg.poses])
        self.get_logger().info(f"Received {len(boxes)} boxes")
        dets = {"dets": [box for box in boxes], "info": [None for _ in boxes]}

        processed_boxes, preds = self.tracker.track(dets)

        if len(processed_boxes) > 0:
            # Process the bounding boxes with the tracker
            self.get_logger().info(f"Processed {len(processed_boxes)} bounding boxes")
            
            boxes = []
            colors = []
            for box in processed_boxes:
                # box is in the format [h, w, l, x, y, z, theta]
                boxes.append([box[X_IDX], box[Z_IDX]])
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
            rviz_marker = detections_to_rviz_marker(
                boxes, poses_msg.header.stamp, poses_msg.header.frame_id, colors=colors)
        else:
            self.get_logger().info("No boxes processed")
            rviz_marker = detections_to_rviz_marker([], poses_msg.header.stamp, poses_msg.header.frame_id)

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
