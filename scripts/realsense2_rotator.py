import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import cv2
import numpy as np

class WayfindingRotationNode(Node):
    def __init__(self, pointcloud=False):
        super().__init__('wayfinding_rotation_node')
        self.get_logger().info("Initializing Wayfinding Rotation Node")
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.depth_subscription # prevent unused variable warning
        self.depth_publisher = self.create_publisher(Image, 'wayfinding/camera/depth/image', 10)
        self.infra_subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.infra_callback,
            10)
        self.infra_subscription # prevent unused variable warning
        self.infra_publisher = self.create_publisher(Image, 'wayfinding/camera/color/image', 10)
        if pointcloud:
            self.pointcloud_subscription = self.create_subscription(
                PointCloud2,
                '/camera/depth/color/points',
                self.pointcloud_callback,
                10)
            self.pointcloud_subscription # prevent unused variable warning
            self.pointcloud_publisher = self.create_publisher(PointCloud2, 'wayfinding/camera/pointcloud', 10)
            self.get_logger().info("Point cloud processing enabled")
        else:
            self.get_logger().info("Point cloud processing disabled")
        self.bridge = CvBridge()

    def depth_callback(self, image_msg):
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        rotated_cvImg = cv2.rotate(cvImg, cv2.ROTATE_90_CLOCKWISE)
        
        # h,w = rotated_cvImg.shape[:2]
        # w_start = (w-CROP_W) // 2
        # h_start = (h-CROP_H) // 2
        # cropped_cvImg = rotated_cvImg[h_start:h_start+CROP_H, w_start:w_start+CROP_W]

        cvImg32F = rotated_cvImg.astype('float32') / 1000.0
        convertedImageMsg = self.bridge.cv2_to_imgmsg(cvImg32F)
        convertedImageMsg.header = image_msg.header
        convertedImageMsg.height = image_msg.width
        convertedImageMsg.width = image_msg.height
        self.depth_publisher.publish(convertedImageMsg)
        # print("Depth callback called")
    
    def infra_callback(self, image_msg):
        """
        Resolves the error: "cv_bridge exception: [8UC1] is not a color format. but [bgr8] is."
        which arises when processing mono images from realsense using opencv.
        Note this is not required for using spencer tracking
        """
        cvImg = self.bridge.imgmsg_to_cv2(image_msg)
        rotated_cvImg = cv2.rotate(cvImg, cv2.ROTATE_90_CLOCKWISE)

        # h,w = rotated_cvImg.shape[:2]
        # w_start = (w-CROP_W) // 2
        # h_start = (h-CROP_H) // 2
        # cropped_cvImg = rotated_cvImg[h_start:h_start+CROP_H, w_start:w_start+CROP_W]
        
        # if len(cropped_cvImg.shape) == 2:
        #     encoding = 'mono8'
        # elif cropped_cvImg.shape[2] == 3:
        encoding = 'bgr8' if image_msg.encoding.lower() in ['bgr8', 'bgr'] else 'rgb8'
        convertedImageMsg = self.bridge.cv2_to_imgmsg(rotated_cvImg, encoding=encoding)
        convertedImageMsg.header = image_msg.header
        convertedImageMsg.height = image_msg.width
        convertedImageMsg.width = image_msg.height
        # convertedImageMsg.encoding = "mono8"
        self.infra_publisher.publish(convertedImageMsg)
        # print("Infra callback called")
    def pointcloud_callback(self, pointcloud_msg):
        # Rotate the point cloud by 90 degrees clockwise
        points = pc2.read_points_numpy(pointcloud_msg, skip_nans=True)
        if points.shape[0] == 0:
            return  # No points to process
        # 90 deg clockwise rotation around Z: (x, y, z) -> (y, -x, z)
        rotated_points_array = np.empty_like(points)
        rotated_points_array[:, 0] = -points[:, 1]   # x' = -y
        rotated_points_array[:, 1] = points[:, 0]   # y' = x
        rotated_points_array[:, 2] = points[:, 2]   # z' = z
        rotated_points_array[:, 3] = points[:, 3]   # rgb = rgb
        rotated_points = rotated_points_array.tolist()

        # Create a new PointCloud2 message with the rotated points
        rotated_pointcloud_msg = pc2.create_cloud(pointcloud_msg.header, pointcloud_msg.fields, rotated_points)
        self.pointcloud_publisher.publish(rotated_pointcloud_msg)

def main(args=None, pointcloud=False):
    rclpy.init(args=args)

    node = WayfindingRotationNode(pointcloud) # Set to True if you want to process point clouds

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Wayfinding Rotation Node.')
    parser.add_argument('--pointcloud', action='store_true', help='Enable point cloud processing')
    args = parser.parse_args()
    main(pointcloud=args.pointcloud)
