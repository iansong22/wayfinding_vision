FROM osrf/ros:noetic-desktop-full

# install ros package
RUN apt-get update && apt-get install -y \
      ros-${ROS_DISTRO}-ros-tutorials \
      ros-${ROS_DISTRO}-common-tutorials && \
    rm -rf /var/lib/apt/lists/*
    
ENV ROS2_DISTRO=foxy

RUN sudo apt update && sudo apt install -y \
    curl gnupg2 lsb-release

RUN apt-get update && apt-get install -y \
    software-properties-common

RUN add-apt-repository universe
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list
# RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Foxy
RUN apt-get update && apt-get install -y \
    ros-${ROS2_DISTRO}-desktop python3-argcomplete \ 
    python3-colcon-common-extensions

# RUN sudo apt-get install -y python-rosdep
# RUN sudo rosdep init
RUN rosdep update
# RUN rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-5.3.1 urdfdom_headers"

ENV COLCON_WS=/ros_workspace
RUN mkdir -p $COLCON_WS/src
# Setup workspace
WORKDIR $COLCON_WS

RUN apt-get install -y git

# RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'
# RUN /bin/bash -c 'source /opt/ros/foxy/setup.bash && colcon build'


WORKDIR $COLCON_WS/src

RUN git clone -b $ROS2_DISTRO https://github.com/ros2/ros1_bridge.git
WORKDIR $COLCON_WS/src/ros1_bridge

# RUN apt-get update && apt-get install -y \
#     ros-foxy-rmw-cyclonedds-cpp \
#     ros-foxy-demo-nodes-cpp \
#     ros-foxy-demo-nodes-py && rm -rf /var/lib/apt/lists/*
# ENV RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
RUN /bin/bash -c "colcon build --symlink-install --packages-skip ros1_bridge"

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash"


RUN /bin/bash -c "colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure"

# Entrypoint
CMD ["/bin/bash"]
