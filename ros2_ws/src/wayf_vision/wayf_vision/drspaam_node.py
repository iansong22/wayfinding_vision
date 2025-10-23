# import time
import time
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

from dr_spaam.detector import Detector

from ament_index_python.packages import get_package_share_directory
import os



class DrSpaamROS(Node):
    """ROS2 node to detect pedestrian using DROW3 or DR-SPAAM."""

    def __init__(self):
        super().__init__("DRSpaam")
        self._read_params()
        self.get_logger().info(f"Starting DR-SPAAM with model path {self.weight_file} and conf_thresh {self.conf_thresh}")
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=True,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        package_name = "wayf_vision"
        package_path = get_package_share_directory(package_name)
        self.declare_parameter("weight_file", os.path.join(package_path, "models/dr_spaam_5_on_frog.pth")) # Default weight file
        self.declare_parameter("conf_thresh", 0.3) # Default confidence threshold
        self.declare_parameter("stride", 1) # Default stride
        self.declare_parameter("detector_model", "DR-SPAAM") # Default detector model
        self.declare_parameter("panoramic_scan", True) # Default panoramic scan setting

        self.weight_file = self.get_parameter("weight_file").get_parameter_value().string_value
        self.conf_thresh = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.stride = self.get_parameter("stride").get_parameter_value().integer_value
        self.detector_model = self.get_parameter("detector_model").get_parameter_value().string_value
        self.panoramic_scan = self.get_parameter("panoramic_scan").get_parameter_value().bool_value

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = self.read_publisher_param("detections")
        self._dets_pub = self.create_publisher(
            PoseArray, topic, queue_size
        )

        topic, queue_size, latch = self.read_publisher_param("rviz")
        self._rviz_pub = self.create_publisher(
            Marker, topic, queue_size
        )
        self._confidence_pub = self.create_publisher(
            MarkerArray, topic + "_confidence", queue_size
        )

        # Subscriber
        topic, queue_size = self.read_subscriber_param("scan")
        self._scan_sub = self.create_subscription(
            LaserScan, topic, self._scan_callback, queue_size
        )
        self._scan_sub

    def _scan_callback(self, msg):
        # if (
        #     self._dets_pub.get_num_connections() == 0
        #     and self._rviz_pub.get_num_connections() == 0
        # ):
        #     return

        # TODO check the computation here
        if not self._detector.is_ready():
            self._detector.set_laser_fov(
                np.rad2deg(msg.angle_increment * len(msg.ranges))
            )

        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        t = time.time()
        dets_xy, dets_cls, _ = self._detector(scan)
        self.get_logger().info(f"End-to-end inference time: {time.time() - t:.4f}s")

        # confidence threshold
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        # self.get_logger().info(
        #     f"Detected {dets_xy}"
        # )

        dets_cls = dets_cls[conf_mask]

        # convert to ros msg and publish
        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

        confidence_msg = detections_to_confidence_array(dets_xy, dets_cls, msg.header)
        self._confidence_pub.publish(confidence_msg)
        

    def read_subscriber_param(self, name):
        """
        @brief      Convenience function to read subscriber parameter.
        """
        # topic = self.get_parameter("~subscriber/" + name + "/topic")
        # queue_size = self.get_parameter("~subscriber/" + name + "/queue_size")
        if name == "scan":
            topic = "/scan"
            queue_size = 1
        return topic, queue_size


    def read_publisher_param(self, name):
        """
        @brief      Convenience function to read publisher parameter.
        """
        # topic = self.get_parameter("~publisher/" + name + "/topic")
        # queue_size = self.get_parameter("~publisher/" + name + "/queue_size")
        # latch = self.get_parameter("~publisher/" + name + "/latch")
        if name == "detections":
            topic = "/dr_spaam_detections"
            queue_size = 1
            latch = False
        elif name == "rviz":
            topic = "/dr_spaam_rviz"
            queue_size = 1
            latch = False
        else:
            raise ValueError(f"Unknown publisher name: {name}")

        return topic, queue_size, latch


def detections_to_rviz_marker(dets_xy, dets_cls):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros"
    msg.id = 0
    msg.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    # red color
    msg.color.r = 1.0
    msg.color.a = 1.0

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy, d_cls in zip(dets_xy, dets_cls):
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

def detections_to_confidence_array(dets_xy, dets_cls, header):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = MarkerArray()

    # to msg
    for i, (d_xy, d_cls) in enumerate(zip(dets_xy, dets_cls)):
        marker = Marker()
        marker.action = Marker.ADD
        marker.ns = "dr_spaam_ros"
        marker.id = i * 1000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.header = header

        # set quaternion so that RViz does not give warning
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.pose.position.x = float(d_xy[0])
        marker.pose.position.y = float(d_xy[1])
        marker.pose.position.z = 1.0  # Slightly above ground

        marker.scale.z = 0.3  # Text height

        # Color gradient from red (low confidence) to green (high confidence)
        marker.color.r = float(1.0 - d_cls)
        marker.color.g = float(d_cls)
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.text = f"{d_cls:.2f}"  # Display confidence value

        msg.markers.append(marker)
            

    return msg

def detections_to_pose_array(dets_xy, dets_cls):
    pose_array = PoseArray()
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()
        p.position.x = float(d_xy[0])
        p.position.y = float(d_xy[1])
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array


def main(args=None):
    rclpy.init(args=args)
    drspaam_node = DrSpaamROS()

    rclpy.spin(drspaam_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
