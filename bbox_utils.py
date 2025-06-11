from visualization_msgs.msg import Marker, MarkerArray
import cv2

def create_bbox3d(x, y, z, width, height, depth, frame_id, timestamp):
    """
    Create a bounding box marker array for each edge for visualization.
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = x + width / 2.0
    marker.pose.position.y = y + height / 2.0
    marker.pose.position.z = z + depth / 2.0
    marker.scale.x = width
    marker.scale.y = height
    marker.scale.z = depth
    marker.color.a = 0.75  # Transparency
    marker.color.r = 0.0  # Red color
    marker.color.g = 1.0
    marker.color.b = 0.0
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
    for i, (x, y, z, width, height, depth) in enumerate(bboxes3d):
        marker = create_bbox3d(x, y, z, width, height, depth, frame_id, timestamp)
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