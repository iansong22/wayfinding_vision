import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

from kalman.model import AB3DMOT as tracker

class KalmanTrackingNode(Node):
    def __init__(self, namespace="camera"):
        super().__init__('kalman_tracking_node')
        self.get_logger().info("Initializing Kalman Tracking Node")
        self.bbox_subscription = self.create_subscription(
            Float32MultiArray,
            namespace + '/yolo/bbox3d',
            self.box_callback,
            10)
        self.bbox_subscription # prevent unused variable warning
        self.tracker = tracker()
    def box_callback(self, bboxes_msg):
        # boxes are in form [x_min, y_min, z_min, x_max, y_max, z_max]
        bboxes = np.array(bboxes_msg.data).reshape(-1, 6)
        self.get_logger().info(f"Received {len(bboxes)} bounding boxes")
        if len(bboxes) > 0:
            # Process the bounding boxes with the tracker
            dets = [{"dets": box, "info": {}} for box in bboxes]
            processed_boxes = self.tracker.track(dets)
            self.get_logger().info(f"Processed {len(processed_boxes)} bounding boxes")
            self.get_logger().info(f"Processed boxes: {processed_boxes}")

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
