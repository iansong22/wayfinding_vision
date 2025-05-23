#!/bin/bash

# This script facilitates running the Docker images after they have been built using the provided Makefile.
#
# By default, X forwarding will be enabled, the nvidia runtime will be used (if available), and the network
# will be isolated from the host.

# Initialize default arguments
COMMAND="tmux"  # what to run once inside the container

# Parse optional command-line arguments
USE_HOST_NETWORK=1
EXTRA_DOCKER_ARGS="$@"

# Build network args
if [ $USE_HOST_NETWORK -eq 1 ]
then
    # This currently requires privileged access, otherwise rviz will crash. See:
    # https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/?answer=301062#post-id-301062
    NETWORK_ARGS="--network=host --privileged --ipc=host --pid=host";
    echo "connection to host network enabled, running in privileged mode"
else
    NETWORK_ARGS="";
    echo "connection to host network disabled, network will be isolated from host. use -h to change this"
fi

# Check if nvidia runtime is available
if docker info 2> /dev/null | grep -i "Runtimes:" | grep "nvidia" > /dev/null
then
  echo "nvidia docker runtime found, CUDA support enabled"
  RUNTIME_ARGS="--runtime=nvidia"
else
  echo "nvidia docker runtime not found, running container without CUDA support"
  RUNTIME_ARGS=""
fi

# Following lines are based upon code from the ROS wiki
# They ensure that X server can be used for graphical applications, e.g. rviz
# See http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration#nvidia-docker2
# Copyright 2020 Open Source Robotics Foundation (CC-BY 3.0 license)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# Run the Docker container in interactive mode
echo "starting container..."
docker run --rm -it \
            --env="DISPLAY=$DISPLAY" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
            $RUNTIME_ARGS \
            $NETWORK_ARGS \
            $EXTRA_DOCKER_ARGS \
            --env="QT_X11_NO_MITSHM=1" \
            --env="XAUTHORITY=$XAUTH" \
            --volume="$XAUTH:$XAUTH" \
            --volume="./logs:/root/catkin_ws/logs" \
            dr_spaam:latest \
            $COMMAND
