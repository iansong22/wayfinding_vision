# ROS base image without GPU support
FROM ros:noetic-perception

# RUN apt-get update && apt-get install -y lsb-release curl && rm -rf /var/lib/apt/lists/*

# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install build tools
RUN apt-get update && apt-get install -y \
      python3-catkin-tools python3-rosdep python3-pip tmux vim \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install osrf-pycommon

# Setup workspace
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src

# Acquire source
#RUN git clone -b $ROS_DISTRO https://github.com/spencer-project/spencer_people_tracking.git
WORKDIR $CATKIN_WS
COPY src src
WORKDIR $CATKIN_WS/src

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
      wget && \
    rosdep update && \
    rosdep install -y -r --from-paths . --ignore-src --rosdistro ${ROS_DISTRO} --as-root=apt:false && \
    rm -rf /var/lib/apt/lists/*

# Build repo
WORKDIR $CATKIN_WS
ENV TERM=xterm
ENV PYTHONIOENCODING=UTF-8
RUN catkin config --extend /opt/ros/$ROS_DISTRO && \
    catkin build -c -s

RUN /bin/bash -c "ln -s /usr/bin/python3 /usr/bin/python"
 # Initialize bashrc
RUN echo "source ~/catkin_ws/devel/setup.bash" >> /root/.bashrc