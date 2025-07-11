from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import cv2

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

def create_bbox3d_array(bboxes3d, frame_id, timestamp):
    """
    Create a MarkerArray containing bounding boxes.
    :param bboxes: List of bounding boxes, each defined as (x, y, z, width, height, depth).
    :param frame_id: The frame ID for the markers.
    :param timestamp: The timestamp for the markers.
    :return: MarkerArray containing all bounding boxes.
    """
    marker_array = MarkerArray()
    for i, bbox_3d in enumerate(bboxes3d):
        marker = create_bbox3d(bbox_3d, frame_id, timestamp)
        marker.id = i  # Unique ID for each marker
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