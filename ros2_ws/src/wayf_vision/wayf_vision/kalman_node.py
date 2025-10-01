import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from wayf_vision.bbox_utils import tracks_to_rviz_marker_array

import numpy as np

from wayf_vision.kalman.model import Wayfinding_3DMOT as tracker

class KalmanTrackingNode(Node):
    def __init__(self):
        super().__init__('kalman_tracking_node')
        self.declare_parameter("namespace", '/camera')
        self.declare_parameter("output_preds", False)
        namespace = self.get_parameter("namespace").get_parameter_value().string_value
        self.output_preds = self.get_parameter("output_preds").get_parameter_value().bool_value

        self.declare_parameter("vis_thres", -0.5)
        self.declare_parameter("lidar_thres", -0.5)
        self.declare_parameter("max_age", 10)
        vis_thres = self.get_parameter("vis_thres").get_parameter_value().double_value
        lidar_thres = self.get_parameter("lidar_thres").get_parameter_value().double_value
        max_age = self.get_parameter("max_age").get_parameter_value().integer_value


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
        self.human_publisher = self.create_publisher(
            MarkerArray,
            namespace + '/kalman/human_tracks',
            10)
        self.bbox_publisher = self.create_publisher(
            MarkerArray,
            namespace + '/kalman/all_tracks',
            10)
        self.tracker = tracker(vis_thres=vis_thres, lidar_thres=lidar_thres, max_age=max_age, output_preds=self.output_preds)
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
        CLASS_IDX = 8
        results = self.tracker.track(dets)
        processed_boxes = results["results"]
        affi = results["affi"]

        if self.tracker.output_preds:
            preds = results["preds"]
            self.get_logger().info(f"Preds: {preds}")
            
        self.get_logger().info(f"affi: {affi}")
            
        tracks = []
        humans = []
        human_colors = []
        colors = []
        
        # Process the bounding boxes with the tracker
        # self.get_logger().info(f"Processed {len(processed_boxes)} bounding boxes: {processed_boxes}")
        for box in processed_boxes:
            # box is in the format [h, w, l, x, y, z, theta]
            idx = int(box[ID_IDX])
            if len(self.colors) > idx:
                color = self.colors[idx]
            elif self.generate_new_colors:
                color = [np.random.rand() for _ in range(3)]
                self.colors.append(color)
            else:
                color = self.colors[idx % len(self.colors)]
            if box[CLASS_IDX] == 1:
                tracks.append(([box[X_IDX], box[Y_IDX]], idx, "Human"))
                humans.append(([box[X_IDX], box[Y_IDX]], idx, "Human"))
                human_colors.append(color)
            else:
                tracks.append(([box[X_IDX], box[Y_IDX]], idx, "Object"))
            colors.append(color)

        # Convert processed boxes to PoseArray for publishing
        rviz_marker = tracks_to_rviz_marker_array(
            tracks, timestamp, frame_id, 0.2, colors=colors)
        human_rviz_marker = tracks_to_rviz_marker_array(
            humans, timestamp, frame_id, 0.2, colors=human_colors)
        
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
        if self.previous_markers_len > len(human_rviz_marker.markers):
            for i in range(len(human_rviz_marker.markers), self.previous_markers_len):
                delete_marker = Marker()
                delete_marker.action = Marker.DELETE
                delete_marker.ns = "yolo_ros"
                delete_marker.id = i
                delete_marker.header.frame_id = frame_id
                delete_marker.header.stamp = timestamp
                human_rviz_marker.markers.append(delete_marker)
        self.previous_markers_len = markers_len

        self.human_publisher.publish(human_rviz_marker)
        self.bbox_publisher.publish(rviz_marker)
                
def main(args=None):
    rclpy.init(args=args)

    node = KalmanTrackingNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
