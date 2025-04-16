FROM ros:humble-ros1-bridge

RUN rm /bin/sh && ln -s /bin/bash /bin/sh 
RUN apt-get update && apt-get install -y \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-cyclonedds

# Alias for sourcing
# sr1 -> source ros 1, sr2 -> source ros 2
RUN echo "alias sr1='source /opt/ros/noetic/setup.bash'" >> ~/.bashrc
RUN echo "alias sr2='source /opt/ros/foxy/setup.bash'" >> ~/.bashrc
ENV ROS_MASTER_URI="http://localhost:11311"

# Example building ROS 1 workspace in dockerfile
# RUN mkdir -p ~/ros1_ws/src
# RUN cd ~/ros1_ws; source /opt/ros/noetic/setup.bash; catkin_make

# Example building ROS 2 workspace in dockerfile
# RUN mkdir -p ~/ros2_ws/src
# RUN cd ~/ros2_ws; source /opt/ros/foxy/setup.bash; colcon build
# ENV ROS_TRANSPORT_OVERRIDES=udp
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

CMD ["/bin/bash"]
