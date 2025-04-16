### To run ros2 <-> ros1 humble bridge:
```
cd ros-humble-ros1-bridge-builder
docker build . -t ros-humble-ros1-bridge-builder
docker run --rm ros-humble-ros1-bridge-builder | tar xvzf -
source ./ros-humble-ros1-bridge/install/local_setup.bash 
ros2 run ros1_bridge dynamic_bridge 
```