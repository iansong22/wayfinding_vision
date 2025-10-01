# wayfinding_vision

Installing requirements and DR-SPAAM:

```
git submodule update --init
pip install -r requirements.txt
cd 2D_lidar_person_detection/dr_spaam
sed -i 's/\.astype(np\.int)/.astype(np.int32)/g' ./dr_spaam/utils/utils.py
python setup.py install
```

Make sure to have ROS2 and cv_bridge for your ROS2 version installed.


To launch:
```
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch wayf_vision wayf_vision_launch.py
```

Running individual nodes:
```
ros2 run wayf_vision kalman_node --ros-args \
  -p namespace:=/camera \
  -p output_preds:=false \
  -p vis_thres:=-0.5 \
  -p lidar_thres:=-0.5 \
  -p max_age:=10 
```
```
ros2 run wayf_vision yolo_node --ros-args \
  -p namespace:=/camera \
  -p rotate:=true \
  -p readCamearInfo:=true\
  -p base_id:laser
```
```
ros2 run wayf_vision drspaam_node --ros-args \
  -p conf_thresh:=0.8
```