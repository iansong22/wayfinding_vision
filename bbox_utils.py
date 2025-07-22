from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseArray
import cv2
import open3d as o3d
import numpy as np

def create_bbox3d(bbox, frame_id, timestamp):
    """
    Create a bounding box marker array for each edge for visualization.
    """
    marker = Marker()
    marker.action = Marker.ADD
    marker.ns = "wayfinding"
    marker.id = 0
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp
    marker.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.03  # line width
    marker.color.a = 0.75  # Transparency
    marker.color.r = 0.0  # Red color
    marker.color.g = 1.0
    marker.color.b = 0.0

    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    for x, y, z in [
        # 12 edges of the bounding box
        (min_x, min_y, min_z), (max_x, min_y, min_z),
        (min_x, min_y, min_z), (min_x, max_y, min_z),
        (min_x, min_y, min_z), (min_x, min_y, max_z),

        (max_x, max_y, min_z), (max_x, min_y, min_z),
        (max_x, max_y, min_z), (max_x, max_y, max_z),
        (max_x, max_y, min_z), (min_x, max_y, min_z),

        (min_x, max_y, max_z), (max_x, max_y, max_z),
        (min_x, max_y, max_z), (min_x, min_y, max_z),
        (min_x, max_y, max_z), (min_x, max_y, min_z),

        (max_x, min_y, max_z), (max_x, max_y, max_z),
        (max_x, min_y, max_z), (max_x, min_y, min_z),
        (max_x, min_y, max_z), (min_x, min_y, max_z)
    ]:
        marker.points.append(Point(x=x, y=y, z=z))
    
    return marker

def create_bbox3d_array(bboxes3d, frame_id, timestamp, num_prev=0):
    """
    Create a MarkerArray containing bounding boxes.
    :param bboxes: List of bounding boxes, each defined as (x, y, z, width, height, depth).
    :param frame_id: The frame ID for the markers.
    :param timestamp: The timestamp for the markers.
    :return: MarkerArray containing all bounding boxes.
    """
    marker_array = MarkerArray()
    for i in range(max(num_prev, len(bboxes3d))):
        if i < len(bboxes3d):
            bbox_3d = bboxes3d[i]
            marker = create_bbox3d(bbox_3d, frame_id, timestamp)
            marker.id = i  # Unique ID for each marker
            marker_array.markers.append(marker)
        else: # remove previous markers if they exist
            # Create a marker with DELETE action to remove it
            marker = Marker()
            marker.action = Marker.DELETE
            marker.ns = "wayfinding"
            marker.id = i  # Unique ID for each marker
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker_array.markers.append(marker)
    return marker_array

def draw_bbox(img, bboxes):
    """
    Draw bounding boxes on an image.
    :param img: The image on which to draw the bounding boxes.
    :param bboxes: List of bounding boxes, each defined as (x1, y1, x2, y2).
    :return: Image with bounding boxes drawn.
    """
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def bbox_3d_to_2d(bbox):
    """
    Convert a 3D bounding box to a 2D bounding box.
    :param bbox: A tuple (x, y, z, width, height, depth) representing the 3D bounding box.
    :return: A tuple (x1, y1, x2, y2) representing the 2D bounding box.
    """
    x, y, z, width, height, depth = bbox
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)
    return (x1, y1, x2, y2)

def add_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def detections_to_rviz_marker_array(dets_xy, timestamp, frame_id, colors=None):
    """
    Convert detections to RViz marker array. Each detection is marked as a circle approximated by line segments.
    :param dets_xy: List of detections, each detection is a tuple (x, y).
    :param timestamp: Timestamp for the marker.
    :param frame_id: Frame ID for the marker.
    :param colors: Optional list of colors for each detection. If None, blue will be used.
    :return msg: Marker message for RViz visualization.
    """
    msg = MarkerArray()
    
    for d_idx, d_xy in enumerate(dets_xy):
        marker = detections_to_rviz_marker([d_xy], timestamp, frame_id, marker_id=d_idx, color=colors[d_idx % len(colors)] if colors else None)
        msg.markers.append(marker)
    
    return msg

def detections_to_rviz_marker(dets_xy, timestamp, frame_id, marker_id=0, color=None):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    :param dets_xy: List of detections, each detection is a tuple (x, y).
    :param timestamp: Timestamp for the marker.
    :param frame_id: Frame ID for the marker.
    ;:param marker_id: Unique ID for the marker.
    :param colors: Optional list of colors for each detection.  If None, blue will be used.
    :return msg: Marker message for RViz visualization.
    
    """
    msg = Marker()
    msg.header.frame_id = frame_id
    msg.header.stamp = timestamp
    msg.action = Marker.ADD
    msg.ns = "yolo_ros"
    msg.id = marker_id
    msg.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    msg.color.r = 0.0
    msg.color.g = 0.0
    msg.color.b = 1.0
    if color is not None:
        msg.color.r = color[0]
        msg.color.g = color[1]
        msg.color.b = color[2]
    msg.color.a = 1.0

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy in dets_xy:
        for i in range(len(xy_offsets) - 1):
            # note, y is up/down so set to 0
            # start point of a segment
            p0 = Point()
            p0.x = float(d_xy[0] + xy_offsets[i, 0])
            p0.z = 0.0
            p0.y = float(d_xy[1] + xy_offsets[i, 1])
            msg.points.append(p0)

            # end point
            p1 = Point()
            p1.x = float(d_xy[0] + xy_offsets[i + 1, 0])
            p1.z = 0.0
            p1.y = float(d_xy[1] + xy_offsets[i + 1, 1])
            msg.points.append(p1)

    return msg

def detections_to_pose_array(dets_xy, timestamp, frame_id):
    pose_array = PoseArray()
    pose_array.header.stamp = timestamp
    pose_array.header.frame_id = frame_id
    for d_xy in dets_xy:
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()
        p.position.x = float(d_xy[0])
        p.position.y = float(d_xy[1])
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array

def depth2PointCloud(depth, rgb, depth_scale, clip_distance_max, mask, intrinsics, voxel_size=0.01):
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
    pc = pc.voxel_down_sample(voxel_size=voxel_size)

    return pc